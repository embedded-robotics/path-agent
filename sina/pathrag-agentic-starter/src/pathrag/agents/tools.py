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
import os, json
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

load_dotenv()
logger = get_logger("pathrag.tools")

NUCLEI_THRESHOLD = 5
GRID_SIZE = 3
OVERLAP = 0.20
DEFAULT_TOP_K = 3

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "src/pathrag/pipeline/files"))
(OUTPUT_ROOT / "query").mkdir(parents=True, exist_ok=True)

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
        rows.append({"image": img, "text": prompt, "question_id": i})
    return rows

def roi_agent_describe(patch, question: str):
    """
    REAL VLM call (LLaVA-Med): single-patch ROI description.
    Requires:
      - PATHRAG_IMAGE: path to the full image (or defaults to sample_he.png)
      - LLMED_REPO, LLMED_MODEL: see client
    """
    img_path = os.environ.get("PATHRAG_IMAGE", "sample_he.png")
    crop_paths = save_crops(img_path, [patch.bbox], "artifacts/crops")
    prompt = f"Briefly describe this pathology ROI to help answer: {question}. One sentence."
    qfile = "artifacts/query/roi.jsonl"
    afile = "artifacts/answer/roi.jsonl"
    _write_jsonl(_rows_for_patches(crop_paths, prompt), qfile)

    try:
        client = LlavaMedClient()
        texts = client.ask_batch(qfile, ".", afile)
        text = texts[0] if texts else f"[roi-fallback] {patch.id}"
        return {"useful": True, "description": text}
    except Exception as e:
        return {"useful": True, "description": f"[roi-error] {e}"}

def patch_agent_contribution(patch, question: str, full_captions: list[str]):
    """
    REAL VLM call (LLaVA-Med): explain contribution of this ROI to the answer.
    """
    img_path = os.environ.get("PATHRAG_IMAGE", "sample_he.png")
    crop_paths = save_crops(img_path, [patch.bbox], "artifacts/crops")
    cap = full_captions[0] if full_captions else "general pathology context"
    prompt = f"In ONE sentence, explain how this ROI helps answer: '{question}'. Use this context: {cap}"
    qfile = "artifacts/query/patch.jsonl"
    afile = "artifacts/answer/patch.jsonl"
    _write_jsonl(_rows_for_patches(crop_paths, prompt), qfile)

    try:
        client = LlavaMedClient()
        texts = client.ask_batch(qfile, ".", afile)
        return texts[0] if texts else f"[patch-fallback] {patch.id} via {cap}"
    except Exception as e:
        return f"[patch-error] {e}"

# def patch_agent_contribution(patch: Patch, question: str, full_captions: list[str]) -> str:
    """Explain how this patch contributes to answering the question.

    Provide one focused sentence (ideal for later fusion).
    You can include a brief reference to the most relevant caption.

    Returns:
        A short string (<= 1–2 lines).
    """
    # TODO (real impl):
    #   crop = load_crop(patch.bbox)
    #   return vlm_or_llm_contribution(crop, question, full_captions)
    cap = full_captions[0] if full_captions else "domain context"
    return f"{patch.id} supports the answer via {cap}."

# =====================================
# STAGE 6: QUESTION-AWARE RE-RANK TOPK
# =====================================
def rerank_for_question(
    patches: list[Patch],
    summaries: list[str],
    question: str,
    k: int,
) -> list[int]:
    """Return indices (into `patches`) for the Top-K most relevant summaries.

    Replace this mock with:
      - Embedding similarity (question vs summary), or
      - A cross-encoder scoring model.

    NOTE: Return INDICES (ints), not Patch objects, to avoid state duplication.
    """
    # TODO (real impl):
    #   scores = cross_encoder(question, summaries)
    #   idx = sorted(range(len(summaries)), key=lambda i: scores[i], reverse=True)[:k]
    return list(range(min(k, len(patches))))


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


def _call_gemmamed_for_critique(prompt: str) -> str:
    """
    Hook this into your actual GemmaMed model.

    Examples:
      - HF Inference API
      - vLLM server
      - Local medgemma_run.py with a different mode

    For now, it's a stub you need to implement.
    """
    raise NotImplementedError("Wire this to your GemmaMed deployment for Stage 5.")


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

    # Try GemmaMed first
    try:
        prompt = _build_gemmamed_critique_prompt(
            question=question,
            patch_summaries=patch_summaries,
            roi_desc=roi_desc,
            round_ix=round_ix,
        )
        raw = _call_gemmamed_for_critique(prompt)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        refined = list(patch_summaries)  # default to originals
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

    # ---------- Fallback: old heuristic behavior ----------
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
def fuse_answer(
    question: str,
    label: str,
    chosen: list[tuple[Patch, str]],
    mode: str = "answer",     # "answer" | "description"
) -> str:
    # Extract short inputs for the text LLM: label + per-patch summaries
    captions = [f"{label}: {s}" for _, s in chosen]
    try:
        return _run_medgemma(question, captions)
    except Exception as e:
        # fallback to the existing mock if MedGemma tool isn’t available
        bullets = "\n".join([f"- {p.id} → {s}" for p, s in chosen])
        return f"[mock {mode}] Q: {question}\nlabel={label}\n{bullets}\n{e}"
