"""
Module: agents.tools

Purpose:
- Define small typed domain objects (e.g., Patch) used across LangGraph nodes.
- Provide typed interfaces for tiling, ranking, retrieval, per-patch agents, critique, and final fusion.
- Keep annotations lazy to avoid import cycles and speed up startup.

Notes:
- Annotations are postponed (strings). Use typing.get_type_hints(...) if runtime resolution is needed.
- Prefer clear type hints (List[Patch], Dict[str, Any]) to improve IDE support and team readability.
- Use @dataclass for compact, maintainable domain models.
"""
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Tuple
import base64
import os, json
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from pathrag.utils.logging import get_logger

from pathrag.site_labeler import predict_tissue
from pathrag.retrieval.store import captions_for_label

import os, json
from pathlib import Path
from pathrag.vision.crop import save_crops
from pathrag.vlm.llava_med_client import LlavaMedClient

from openai import OpenAI

load_dotenv()
logger = get_logger("pathrag.tools")

NUCLEI_THRESHOLD = 5
GRID_SIZE = 3
OVERLAP = 0.20
DEFAULT_TOP_K = 3

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "src/pathrag/pipeline/files"))
(OUTPUT_ROOT / "query").mkdir(parents=True, exist_ok=True)

_LAST_IMAGE_PATH: str | None = None


def _get_openai_client() -> OpenAI:
    return OpenAI()


def _post_json(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"Failed POST {url}: {e}") from e


def _encode_image_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _call_stage4_remote(endpoint: str, crop_path: str, prompt: str, extra: dict | None = None) -> str:
    base_url = os.environ.get("PATHRAG_STAGE4_REMOTE_URL", "").rstrip("/")
    if not base_url:
        raise RuntimeError("PATHRAG_STAGE4_REMOTE_URL is not set")
    payload = {
        "image_b64": _encode_image_base64(crop_path),
        "filename": Path(crop_path).name,
        "prompt": prompt,
    }
    if extra:
        payload.update(extra)
    res = _post_json(
        f"{base_url}/{endpoint.lstrip('/')}",
        payload,
        timeout=int(os.getenv("PATHRAG_HTTP_TIMEOUT", "1800")),
    )
    text = str(res.get("text", "")).strip()
    if not text:
        raise RuntimeError(f"Remote Stage 4 returned empty text from {endpoint}")
    return text

import subprocess
from pathlib import Path

def _run_medgemma(question: str, captions: list[str], tool_dir: str | None = None) -> str:
    tool = Path(tool_dir or os.environ.get("MEDGEMMA_TOOL_DIR", "tools/medgemma")).resolve()
    py   = tool / ".venv/bin/python"
    cfg  = tool / "config/default.yaml"
    out  = tool / "artifacts/answer/out.json"

    if not py.exists():
        raise RuntimeError(f"MedGemma tool not found: {py}\nSet MEDGEMMA_TOOL_DIR or install the tool venv.")

    # write a temp captions file next to the tool (no cross-env imports)
    tmp = tool / "data" / "captions_from_graph.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(captions), encoding="utf-8")

    cmd = [
        str(py), "-m", "src.graph.medgemma_run",
        "--config", str(cfg),
        "--question", question,
        "--captions-file", str(tmp),
        "--out", str(out),
        "--k", str(min(len(captions), 6)),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tool))
    if r.returncode != 0:
        raise RuntimeError(f"[MedGemma failed]\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return json.loads(out.read_text()).get("answer", "").strip()


# Lightweight Patch dataclass used throughout the graph. Kept minimal so callers
# can construct via Patch(**p) when p is a dict coming from state.
@dataclass
class Patch:
    id: str
    bbox: Tuple[int, int, int, int]
    score: float = 0.0

# =========================
# STAGE 1: TILING + HC RANK
# =========================
def tile_image(image_path: str, tile_size: int = 224) -> List[Patch]:
    """Split the input image into a regular grid of tiles (no GPU required).

    Design:
      - Keeps IDs deterministic (P0, P1, …) based on scan order.
      - Does NOT load image pixels here if you already have a tiler upstream; feel free to replace
        with your WSI tiler (OpenSlide, tifffile) that yields (bbox) without loading full image.

    Args:
        image_path: Path to the WSI or large image.
        tile_size: Width/height of square tiles (pixels).
        stride: Optional stride; if None, defaults to tile_size (no overlap).
                Use a smaller stride (< tile_size) to introduce overlap.

    Returns:
        A dense list of Patch with zero scores (to be ranked in the next step).
    """
    global _LAST_IMAGE_PATH
    _LAST_IMAGE_PATH = image_path
    # TODO (real impl):
    #   - Use OpenSlide/pyvips/tifffile to stream tiles without loading the full WSI.
    #   - Compute (W, H) from image metadata; iterate y,x over range(0, H, stride).
    # MOCK (minimal): return 9 tiles of 224×224 from the top-left quadrant
    tiles: List[Patch] = []
    k = 0
    grid = 3  # 3×3 as a lightweight demo
    step = tile_size
    for i in range(grid):
        for j in range(grid):
            bbox = (j*step, i*step, j*step + tile_size, i*step + tile_size)
            tiles.append(Patch(id=f"P{k}", bbox=bbox, score=0.0))
            k += 1
    logger.info(f"Tiled image {image_path} into {len(tiles)} patches.")
    return tiles


def histocartography_rank(image_path: str, patches: List[Patch]) -> List[Patch]:
    """Rank patches by a HistoCartography-derived proxy (e.g., nuclei density).

    Contract:
      - SAME length/order as input is NOT required; we return a sorted list (desc by score).
      - Score meaning: larger == more informative (for H&E).

    Replace this mock with:
      - Your nuclei segmentation + counting per patch, or
      - Any HC embedding → importance scorer (e.g., graph centrality).

    Args:
        image_path: Path to image (used if you crop & analyze pixels here).
        patches: Dense grid from `tile_image`.

    Returns:
        New list of Patch with `.score` filled, sorted descending by score.
    """
    # TODO (real impl):
    #   for p in patches:
    #       crop = read_crop(image_path, p.bbox)
    #       score = nuclei_count(crop) or nuclei_density(crop)
    #       out.append(Patch(p.id, p.bbox, float(score)))
    # MOCK: monotonically decreasing scores
    ranked: List[Patch] = []
    for i, p in enumerate(patches):
        ranked.append(Patch(id=p.id, bbox=p.bbox, score=1.0 - 0.05 * i))
    ranked.sort(key=lambda q: q.score, reverse=True)
    logger.info(f"Ranked {len(ranked)} patches by HistoCartography proxy.")
    return ranked

# ==============================
# STAGE 2: CHEIF + COMMON PATCH
# ==============================
def cheif_rank(image_path: str, patches: List[Patch]) -> List[Patch]:
    """Rank patches by CHEIF attention (foundation model signal).

    Replace this mock with:
      - A call to your CHEIF model to produce per-patch attentions/saliency.
      - Normalize scores to [0,1] if you mix with other sources.

    Returns:
        List[Patch] sorted descending by attention score.
    """
    # TODO (real impl):
    #   att = cheif_attention(image_path, [p.bbox for p in patches]) -> List[float]
    #   return sorted([Patch(p.id, p.bbox, att[i]) ...], key=lambda x: x.score, reverse=True)
    ranked: List[Patch] = []
    for i, p in enumerate(patches):
        ranked.append(Patch(id=p.id, bbox=p.bbox, score=1.0 - 0.03 * i))
    ranked.sort(key=lambda q: q.score, reverse=True)
    return ranked


def common_patches(hc: List[Patch], cheif: List[Patch], top_k: int = 6) -> List[Patch]:
    """Intersect/merge HC and CHEIF rankings into a consensus Top-K.

    Strategy (simple & effective):
      1) Take a wider band from each list (e.g., 2×top_k) to avoid missing near-misses.
      2) Merge by patch.id, accumulate scores from both sources.
      3) Aggregate scores (mean) → sort desc → take Top-K.

    Notes:
      - If HC/CHEIF disagree on bbox (shouldn’t if tiling is shared), prefer the first occurrence.
      - You can switch to rank-based fusion (e.g., Borda count) if score scales differ a lot.

    Args:
        hc: HC-ranked patches (desc by HC score).
        cheif: CHEIF-ranked patches (desc by attention).
        top_k: Output size after fusion.

    Returns:
        Top-K fused Patch list (desc by aggregated score).
    """
    use_combined_api = os.getenv("PATHRAG_USE_COMBINED_API", "0") == "1"
    combined_api_url = os.getenv("PATHRAG_COMBINED_API_URL", "http://localhost:8003/process")
    if use_combined_api and _LAST_IMAGE_PATH and str(_LAST_IMAGE_PATH).lower().endswith(".svs"):
        try:
            payload = {
                "image_path": _LAST_IMAGE_PATH,
                "svs_level": int(os.getenv("PATHRAG_SVS_LEVEL", "2")),
                "top_n": top_k,
                "save_patches": False,
                "patch_size": int(os.getenv("PATHRAG_PATCH_SIZE", "224")),
                "anatomical_label": int(os.getenv("PATHRAG_ANATOMICAL_LABEL", "1")),
                "top_k": top_k,
            }
            res = _post_json(combined_api_url, payload, timeout=int(os.getenv("PATHRAG_HTTP_TIMEOUT", "1800")))
            chief = (res.get("chief_response") or {}).get("top_k_patches") or []
            if chief:
                out: List[Patch] = []
                for i, p in enumerate(chief):
                    bbox = (int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]))
                    out.append(Patch(id=f"CP{i}", bbox=bbox, score=float(len(chief) - i)))
                logger.info(f"Used Combined API patches (k={len(out)})")
                return out[:top_k]
        except Exception as e:
            logger.warning(f"Combined API fallback to local fusion: {e}")

    band = top_k * 2
    by_id: Dict[str, Tuple[Tuple[int,int,int,int], List[float]]] = {}

    def add(lst: List[Patch]):
        for p in lst[:band]:
            if p.id not in by_id:
                by_id[p.id] = (p.bbox, [p.score])
            else:
                by_id[p.id][1].append(p.score)

    add(hc)
    add(cheif)

    fused: List[Patch] = []
    for pid, (bbox, scores) in by_id.items():
        avg = sum(scores) / len(scores)   # simple average of available scores
        fused.append(Patch(id=pid, bbox=bbox, score=avg))

    fused.sort(key=lambda q: q.score, reverse=True)
    logger.info(f"Fused {len(fused)} patches from HC and CHEIF into Top-{top_k}.")
    return fused[:top_k]

# ===============================
# STAGE 3: LABELING + RETRIEVAL
# ===============================
def identify_subpathology(image_path: str) -> str:
    """Predict a coarse tissue/site/sub-pathology label from the image.

    Replace this mock with your actual classifier (e.g., PLIP/CLIP head, UTSW model).
    Keep the return value SHORT and canonicalized (used as a retrieval query/key).

    Args:
        image_path: path to the source image (or WSI).

    Returns:
        A normalized label string (e.g., "squamous_cell_carcinoma").
    """
    global _LAST_IMAGE_PATH
    _LAST_IMAGE_PATH = image_path
    return predict_tissue(image_path)["label"]  # TODO (real impl):


def retrieve_subpath_captions(label: str, top_m: int = 5) -> list[str]:
    """Fetch top-M textual snippets/captions relevant to the label.

    Replace this mock with your retriever (BiomedCLIP/PLIP embeddings + FAISS).
    Normalize text (strip, de-dupe) and keep snippets concise.

    Args:
        label: normalized label from identify_subpathology.
        top_m: how many captions to return.

    Returns:
        List of short strings (captions/knowledge to condition later agents).
    """
    use_retriever_api = os.getenv("PATHRAG_USE_RETRIEVER_API", "0") == "1"
    retriever_url = os.getenv("PATHRAG_RETRIEVER_API_URL", "http://localhost:8000/retrieve_captions")
    if use_retriever_api and _LAST_IMAGE_PATH:
        try:
            payload = {
                "images": [_LAST_IMAGE_PATH],
                "anatomical_site": label,
                "k": top_m,
            }
            res = _post_json(retriever_url, payload, timeout=int(os.getenv("PATHRAG_HTTP_TIMEOUT", "120")))
            # Endpoint returns List[List[Dict]]
            first = res[0] if isinstance(res, list) and res else []
            caps = [str(r.get("caption", "")).strip() for r in first if isinstance(r, dict) and r.get("caption")]
            if caps:
                return caps[:top_m]
        except Exception as e:
            logger.warning(f"Retriever API fallback to local bank: {e}")
    return captions_for_label(label, top_m=top_m)    # TODO (real impl):


# =======================================
# STAGE 4: ROI + PATCH AGENTS (per patch)
# =======================================
def _write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _rows_for_patches(patch_image_paths, prompt):
    # LLaVA expects: {"image": "<file>", "text": "<prompt>", "question_id": i}
    rows = []
    for i, img in enumerate(patch_image_paths):
        abs_img = str(Path(img).resolve())   # <-- make image path absolute
        rows.append(
            {
                "image": abs_img,
                "text": prompt,
                "question_id": i,
            }
        )
    return rows

def roi_agent_describe(patch, question: str, image_path: str | None = None):
    """
    REAL VLM call (LLaVA-Med): single-patch ROI description.
    Requires:
      - PATHRAG_IMAGE: path to the full image (or defaults to sample_he.png)
      - LLMED_REPO, LLMED_MODEL: see LlavaMedClient
    """
    img_path = image_path or os.environ.get("PATHRAG_IMAGE", "sample_he.png")
    crop_paths = save_crops(img_path, [patch.bbox], "artifacts/crops")

    prompt = (
        f"Question: {question}\n"
        "Describe the dominant visible structure in this crop using only morphology. "
        "State whether it looks more like a blood vessel, a duct or gland, or another structure, "
        "and justify that with visible features such as wall thickness, lumen, lining, shape, and surrounding stroma. "
        "A thick wall favors blood vessel over cystic space. "
        "Do not call it cystic unless the wall is thin and the morphology clearly supports that interpretation. "
        "Do not give a disease diagnosis.\n"
        "Return exactly 1 short sentence."
    )

    qfile = Path("artifacts/query/roi.jsonl").resolve()
    afile = Path("artifacts/answer/roi.jsonl").resolve()
    _write_jsonl(_rows_for_patches(crop_paths, prompt), str(qfile))
    crop_path = crop_paths[0]

    try:
        if os.environ.get("PATHRAG_STAGE4_REMOTE_URL"):
            text = _call_stage4_remote("describe_roi", crop_path, prompt)
            return {"useful": True, "description": text}
        client = LlavaMedClient()
        texts = client.ask_batch(str(qfile), ".", str(afile))
        text = texts[0] if texts else f"[roi-fallback] {patch.id}"
        return {"useful": True, "description": text}
    except Exception as e:
        return {"useful": False, "description": f"[roi-error] {e}"}

def patch_agent_contribution(
    patch: Patch,
    question: str,
    full_captions: list[str],
    image_path: str | None = None,
) -> str:
    """
    REAL VLM call (LLaVA-Med): explain contribution of this ROI to the answer.
    Returns a short sentence, used in Stage 5/7 fusion.
    """
    img_path = image_path or os.environ.get("PATHRAG_IMAGE", "sample_he.png")
    crop_paths = save_crops(img_path, [patch.bbox], "artifacts/crops")

    use_stage3_captions = os.getenv("PATHRAG_STAGE4_USE_CAPTIONS", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    cap = (full_captions[0] if (use_stage3_captions and full_captions) else "").strip()
    if cap:
        cap = cap[:240]

    prompt_parts = [
        f"Question: {question}",
        "Decide whether the dominant structure in this crop is more consistent with a blood vessel, "
        "a duct or gland, or is uncertain, using only visible morphology. "
        "Use evidence such as wall thickness, lumen, lining, shape, and surrounding stroma. "
        "Do not name a disease diagnosis.",
    ]
    if cap:
        prompt_parts.append("Ignore the context if it is not visibly supported.")
        prompt_parts.append(f"Context: {cap}")
    prompt_parts.append(
        "Return exactly 1 short sentence that starts with one of these labels: "
        "Direct evidence:, Partial evidence:, or No useful evidence:. "
        "If relevant, explicitly say whether the morphology favors blood vessel, duct or gland, or remains uncertain."
    )
    prompt = "\n".join(prompt_parts)
    fallback_prompt = (
        f"Question: {question}\n"
        "Look only at this crop. State whether it provides direct evidence, partial evidence, or no useful evidence "
        "for answering the question. Mention only visible morphology such as lumen, wall thickness, lining, shape, "
        "and surrounding stroma, and say whether the structure favors blood vessel, duct or gland, or is uncertain. "
        "Keep it to 1 short sentence.\n"
        "Return exactly 1 short sentence starting with Direct evidence:, Partial evidence:, or No useful evidence:."
    )

    qfile = Path("artifacts/query/patch.jsonl").resolve()
    afile = Path("artifacts/answer/patch.jsonl").resolve()
    _write_jsonl(_rows_for_patches(crop_paths, prompt), str(qfile))
    crop_path = crop_paths[0]

    try:
        if os.environ.get("PATHRAG_STAGE4_REMOTE_URL"):
            try:
                return _call_stage4_remote(
                    "patch_contribution",
                    crop_path,
                    prompt,
                    extra={"question": question, "context": cap} if cap else {"question": question},
                )
            except RuntimeError as e:
                if "empty text" not in str(e):
                    raise
                logger.warning(f"Remote patch_contribution returned empty text for {patch.id}; retrying with fallback prompt")
                return _call_stage4_remote(
                    "patch_contribution",
                    crop_path,
                    fallback_prompt,
                    extra={"question": question},
                )
        client = LlavaMedClient()
        texts = client.ask_batch(str(qfile), ".", str(afile))
        return texts[0] if texts else f"[patch-fallback] {patch.id} via {cap}"
    except Exception as e:
        return f"[patch-error] {e}"


@dataclass
class PatchInfo:
    id: str
    bbox: Tuple[int, int, int, int]
    score: float = 0.0


def run_histocartography(image_path: str, top_k: int = 3) -> Dict[str, Any]:
    tiles = tile_image(image_path)
    hc = histocartography_rank(image_path, tiles)
    ch = cheif_rank(image_path, tiles)
    fused = common_patches(hc, ch, top_k=top_k)
    return {
        "tiles": [t.__dict__ for t in tiles],
        "patches": [PatchInfo(id=p.id, bbox=p.bbox, score=p.score).__dict__ for p in fused],
    }


def make_llava_query_files(image_path: str, question: str, patches: List[PatchInfo]) -> Dict[str, str]:
    crop_paths = save_crops(image_path, [p.bbox for p in patches], "artifacts/crops")
    qfile = Path("artifacts/query/patch_questions.jsonl").resolve()
    rows = _rows_for_patches(crop_paths, question)
    _write_jsonl(rows, str(qfile))
    return {"question_jsonl": str(qfile), "crop_dir": str(Path("artifacts/crops").resolve())}


def prepare_and_run(image_path: str, question: str, mode: str = "answer", top_k: int = 3) -> Dict[str, Any]:
    from pathrag.agents.langgraph_app import build_graph

    app = build_graph()
    init = {
        "image_path": image_path,
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "max_rounds": int(os.getenv("PATHRAG_MAX_ROUNDS", "1")),
        "round_ix": 0,
    }
    last = None
    for s in app.stream(init, config={"configurable": {"thread_id": "autogen-run"}}):
        last = s

    if not isinstance(last, dict):
        return {"final_answer": "", "patches": [], "full_captions": []}

    fuse = last.get("fuse", last)
    identify = last.get("identify", last)
    tile = last.get("tile_rank", last)
    return {
        "final_answer": fuse.get("final_answer", last.get("final_answer", "")),
        "patches": tile.get("patches", last.get("patches", [])),
        "full_captions": identify.get("full_captions", last.get("full_captions", [])),
        "subpath_label": identify.get("subpath_label", last.get("subpath_label", "")),
    }

# =====================================
# STAGE 6: QUESTION-AWARE RE-RANK TOPK
# =====================================
def rerank_for_question(
    patches: list[Patch],
    summaries: list[str],
    question: str,
    k: int,
    min_keep: int = 1,
    score_drop_ratio: float = 0.7,
) -> list[int]:
    """Return indices (into `patches`) for the most relevant summaries.

    Stage 6: question-aware re-ranking with *dynamic* Top-K.

    - Uses Stage-5 tags (CENTRAL/SUPPORTING/OFF_TOPIC/CONTRADICTORY) in the summaries.
    - Scores each summary against the question with a tiny lexical overlap.
    - Boosts CENTRAL/SUPPORTING, penalizes OFF_TOPIC/CONTRADICTORY.
    - Keeps all patches whose score is “close enough” to the best score, up to k.
    - Guarantees that at least `min_keep` patches are returned (if any exist).
    """
    n = min(len(patches), len(summaries))
    if n == 0:
        return []

    # If caller passed 0 or negative k, treat it as “no explicit cap”.
    if k <= 0:
        k = n

    scored: list[tuple[int, float]] = []
    for i in range(n):
        text = summaries[i] or ""

        # Base lexical relevance (Stage-5 helper)
        s = _simple_overlap_score(question, text)

        # Boost/penalize based on Stage-5 role tags in the summary header.
        upper = text.upper()
        if "[CENTRAL]" in upper:
            s *= 1.3
        elif "[SUPPORTING]" in upper:
            s *= 1.1
        elif "[CONTRADICTORY]" in upper:
            s *= 0.3
        elif "[OFF_TOPIC]" in upper:
            s *= 0.4

        scored.append((i, s))

    # Sort patches by score (desc)
    scored.sort(key=lambda t: t[1], reverse=True)
    best_score = scored[0][1]

    # If everything scores 0, just return first k indices.
    if best_score <= 0:
        return [idx for idx, _ in scored[: min(k, n)]]

    selected: list[int] = []

    for idx, s in scored:
        # Once we’ve kept at least min_keep, enforce the drop threshold.
        if s < best_score * score_drop_ratio and len(selected) >= min_keep:
            break
        selected.append(idx)
        if len(selected) >= k:
            break

    # Enforce min_keep: pad with next best patches if needed.
    if len(selected) < min_keep:
        for idx, _ in scored:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min_keep or len(selected) >= k:
                break

    # Final safety: clamp to valid range.
    selected = [i for i in selected if 0 <= i < n]
    return selected


# ==========================
# Stage 5 
# ==========================

from typing import List
import math


def _simple_overlap_score(question: str, text: str) -> float:
    """
    Tiny lexical overlap score between question and text.
    Just to have *some* critique signal without calling an LLM.
    """
    if not question or not text:
        return 0.0

    q_tokens = {t.lower() for t in question.split() if len(t) > 2}
    t_tokens = {t.lower() for t in text.split() if len(t) > 2}
    if not q_tokens or not t_tokens:
        return 0.0

    inter = len(q_tokens & t_tokens)
    return inter / math.sqrt(len(q_tokens) * len(t_tokens))

import re

def _build_gemmamed_critique_prompt(
    question: str,
    patch_summaries: List[str],
    roi_desc: List[str],
    round_ix: int,
) -> str:
    """
    Build a single batched prompt for GemmaMed.
    GemmaMed should return one line per patch, e.g.:

        [PATCH=0] [CENTRAL] Rewritten summary...

    We keep it text-only so you can hook it to HF/vLLM/your own server.
    """
    blocks = []
    for i, base in enumerate(patch_summaries):
        roi = roi_desc[i] if i < len(roi_desc) else ""
        blocks.append(
            f"Patch {i}:\n"
            f"Summary: {base or '[empty]'}\n"
            f"ROI: {roi or 'N/A'}"
        )

    patches_block = "\n\n".join(blocks)

    prompt = f"""
You are GemmaMed, a careful pathology VQA assistant.

We are in critique round {round_ix}.

Task:
For EACH patch, you must:
1. Decide if the patch is:
   - CENTRAL: crucial to answer the question.
   - SUPPORTING: helpful but not the main evidence.
   - OFF_TOPIC: mostly irrelevant.
   - CONTRADICTORY: conflicts with the likely correct answer.
2. Rewrite the patch summary in at most 2 sentences, focusing ONLY on details
   relevant to answering the question.

Output format:
Return EXACTLY ONE line per patch, with:

  [PATCH=i] [ROLE] rewritten-summary-here

where ROLE is CENTRAL, SUPPORTING, OFF_TOPIC, or CONTRADICTORY.

Do NOT include any extra commentary.

Question:
{question}

Patches:
{patches_block}
"""
    return prompt.strip()


def _call_gemmamed_for_critique(prompt: str, temperature: float = 0.2) -> str:
    """
    Call GemmaMed (or any chat model you've configured as 'gemma-med')
    with a single prompt string and return the raw text response.
    """
    messages = [
        {"role": "system", "content": "You are GemmaMed, a careful pathology VQA assistant."},
        {"role": "user", "content": prompt},
    ]
    response = _get_openai_client().chat.completions.create(
        model="gpt-4.1-mini",  # or your exact model name
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def critique_round(
    patch_summaries: List[str],
    roi_desc: List[str],
    question: str,
    round_ix: int,
) -> List[str]:
    """
    Stage 5: one critique pass over patch_summaries.

    Preferred path:
      - Use GemmaMed (via _call_gemmamed_for_critique) to classify each patch
        (CENTRAL/SUPPORTING/OFF_TOPIC/CONTRADICTORY) and rewrite the summary.

    Fallback:
      - If GemmaMed is not configured or errors, fall back to the simple lexical
        heuristic using _simple_overlap_score, preserving the old behavior.

    Returns:
      list[str] of the SAME length as patch_summaries, but with enriched tags:
        [ROUND=r] [ROLE] [PATCH=i] rewritten summary...
    """
    if not patch_summaries:
        return []

    # ---- Try GemmaMed first ----
    try:
        prompt = _build_gemmamed_critique_prompt(
            question=question,
            patch_summaries=patch_summaries,
            roi_desc=roi_desc,
            round_ix=round_ix,
        )
        raw = _call_gemmamed_for_critique(prompt)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        refined = list(patch_summaries)  # default: originals
        pattern = re.compile(r"\[PATCH=(\d+)\]\s*\[(\w+)\]\s*(.*)")

        for ln in lines:
            m = pattern.match(ln)
            if not m:
                continue
            idx = int(m.group(1))
            role = m.group(2)
            body = m.group(3).strip()
            if 0 <= idx < len(refined):
                refined[idx] = f"[ROUND={round_ix}] [{role}] [PATCH={idx}] {body}"

        return refined

    except Exception as e:
        logger.warning(f"GemmaMed critique failed, falling back to heuristic: {e}")

    # ---- Fallback: old heuristic behavior ----
    refined: List[str] = []

    n = min(len(patch_summaries), len(roi_desc)) if roi_desc else len(patch_summaries)
    for i in range(n):
        base = patch_summaries[i] or ""
        roi = roi_desc[i] if i < len(roi_desc) else ""

        overlap = _simple_overlap_score(question, base + " " + roi)

        if overlap < 0.05:
            role = "OFF_TOPIC"
        elif overlap > 0.25:
            role = "CENTRAL" if overlap > 0.40 else "SUPPORTING"
        else:
            role = "UNCERTAIN"

        header = f"[ROUND={round_ix}] [{role}] [PATCH={i}]"
        merged_context = base.strip()
        if roi:
            merged_context = f"{merged_context} (ROI: {roi.strip()})".strip()

        refined_text = f"{header} {merged_context}"
        refined.append(refined_text)

    # if roi_desc shorter than patch_summaries, append untouched tails
    for j in range(n, len(patch_summaries)):
        refined.append(patch_summaries[j])

    return refined

def _llm_critique_single(
    question: str, roi: str, summary: str, role_hint: str, round_ix: int
) -> str:
    """
    Wraps an LLM call that rewrites the patch summary.

    Return: a single line of text including tags like [CENTRAL]/[OFF_TOPIC]
    so Stage-6 can still parse it as plain text.
    """
    # Pseudocode — wire to your actual client
    prompt = f"""
    You are a pathology VQA critique agent.

    Question: {question}

    ROI description:
    {roi}

    Patch summary:
    {summary}

    Role hint: {role_hint}
    Round: {round_ix}

    1. Decide whether this patch is CENTRAL, SUPPORTING, or OFF_TOPIC
       for answering the question.
    2. Rewrite the patch summary in at most 2 sentences, focusing only on
       details relevant to answering the question.
    3. Start your answer with a tag in square brackets, one of
       [CENTRAL], [SUPPORTING], or [OFF_TOPIC].

    Return ONLY a single line of text.
    """
    # response = client.responses.create(...)
    # return response.output[0].content[0].text
    raise NotImplementedError("Hook up LLM here")

# ============================
# STAGE 7: FINAL FUSION (LLM)
# ============================
# ============================
# STAGE 7: FINAL FUSION (LLM)
# ============================
def fuse_answer(
    question: str,
    label: str,
    chosen: list[tuple[Patch, str]],
    full_captions: list[str] | None = None,
    mode: str = "answer",  # "answer" | "description"
) -> str:
    """
    Stage 7 fusion helper (GPT-4+ pathologist).

    Inputs:
      - question: original user question.
      - label: sub-pathology label (e.g., "scc").
      - chosen: list of (Patch, summary) after Stage 6 selection.
      - full_captions: optional weak site-context hints from Stage 3.
      - mode: "answer" (VQA-style final answer) or "description" (findings-style text).

    Behavior:
      - Builds a textual evidence block from critiqued patch summaries.
      - Calls GPT-4-class model for final reasoning over this evidence.
      - If GPT call fails, falls back to a debug string with raw evidence.
    """
    if not chosen:
        # No patches survived Stage 6 – be explicit instead of hallucinating.
        return (
            f"[no-evidence] Could not select any informative patches for label "
            f"'{label}' to answer: {question}"
        )

    # Build an evidence block: one line per chosen patch.
    # Summaries already contain Stage-5 tags like [CENTRAL]/[SUPPORTING]/[OFF_TOPIC].
    evidence_lines = []
    for patch, summary in chosen:
        evidence_lines.append(f"[PATCH_ID={patch.id}] {summary}")
    evidence_block = "\n".join(evidence_lines)
    context_lines = [c.strip() for c in (full_captions or []) if str(c).strip()]
    context_block = "\n".join(f"- {c}" for c in context_lines[:5]) if context_lines else "(none)"

    # Adjust the “task” based on Path-RAG's (answer) vs (description) variants.
    if mode == "description":
        # Path-RAG (description): GPT-4 gets descriptions and produces a richer narrative
        task = (
            f"Provide a concise but informative pathology description for this case "
            f"of '{label}', using only the evidence from the selected patches."
        )
    else:
        # Path-RAG (answer): GPT-4 produces a direct answer to the VQA question.
        task = (
            "Answer the pathology question using only the evidence from the selected "
            "patches. If the evidence is insufficient, say that explicitly."
        )

    # Construct the user prompt for GPT-4(+)
    prompt = f"""
You are a professional pathologist.

Task:
{task}

Question:
{question}

Sub-pathology label:
{label}

Weak site-context hints:
{context_block}

Evidence from selected patches:
{evidence_block}

Instructions:
- Treat the selected patch evidence as the primary source of truth.
- Treat the sub-pathology label and site-context hints as weak background context only.
- If the label or hints conflict with visible morphology from the patches, ignore the label or hints.
- Do NOT invent findings that are not supported by the patches.
- If evidence is conflicting or incomplete, acknowledge that.
- Prefer naming visible structures over proposing disease labels when the morphology does not justify a diagnosis.
- Respond in clear, clinical language in one or two paragraphs.
""".strip()

    try:
        # Call a GPT-4-class model (pick whatever you are using: gpt-4o, gpt-4.1, etc.)
        response = _get_openai_client().chat.completions.create(
            model="gpt-4o",  # <--- change this to your preferred GPT model
            messages=[
                {"role": "system", "content": "You are a careful medical AI assistant and expert pathologist."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback: at least expose the evidence so you can debug
        debug_block = "\n".join([f"- {p.id} → {s}" for p, s in chosen])
        return (
            f"[fusion-fallback {mode}] Q: {question}\n"
            f"label={label}\n"
            f"{debug_block}\n"
            f"Error calling GPT model: {e}"
        )
