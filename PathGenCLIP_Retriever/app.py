import gc
import os
from typing import Dict, List, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import open_clip

app = FastAPI(title="Anatomical-Site CLIP Matcher", version="1.0")
# to run:  python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# anatomical_site -> checkpoint path/name
model_names: Dict[str, str] = {
    "adrenal gland": "pathgenclip_models/pathgenclip_best_adrenal_gland.pt",
    "bladder": "pathgenclip_models/pathgenclip_best_bladder.pt",
    "brain": "pathgenclip_models/pathgenclip_best_brain.pt",
    "breast": "pathgenclip_models/pathgenclip_best_breast.pt",
    "cervix": "pathgenclip_models/pathgenclip_best_cervix.pt",
    "colon": "pathgenclip_models/pathgenclip_best_colon.pt",
    "esophagus": "pathgenclip_models/pathgenclip_best_esophagus.pt",
    "kidney": "pathgenclip_models/pathgenclip_best_kidney.pt",
    "liver": "pathgenclip_models/pathgenclip_best_liver.pt",
    "lung": "pathgenclip_models/pathgenclip_best_lung.pt",
    "ovary": "pathgenclip_models/pathgenclip_best_ovary.pt",
    "pancreas": "pathgenclip_models/pathgenclip_best_pancreas.pt",
    "prostate": "pathgenclip_models/pathgenclip_best_prostate.pt",
    "skin": "pathgenclip_models/pathgenclip_best_skin.pt",
    "soft tissue": "pathgenclip_models/pathgenclip_best_soft_tissue.pt",
    "stomach": "pathgenclip_models/pathgenclip_best_stomach.pt",
    "testis": "pathgenclip_models/pathgenclip_best_testis.pt",
    "thyroid": "pathgenclip_models/pathgenclip_best_thyroid.pt",
    "uterus": "pathgenclip_models/pathgenclip_best_uterus.pt" }


DEFAULT_MODEL_ARCH = "ViT-B-16"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class MatchRequest(BaseModel):
    image_path: str = Field(..., description="Local path to the image file")
    queries: List[str] = Field(..., min_length=1)
    anatomical_site: str
    k: int = Field(5, ge=1)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.inference_mode()
def run_clip_from_path(
    image_path: str,
    queries: List[str],
    anatomical_site: str,
) -> List[Tuple[str, float]]:
    if anatomical_site not in model_names:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown anatomical_site '{anatomical_site}'. Available: {sorted(model_names.keys())}",
        )

    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        raise HTTPException(status_code=400, detail=f"image_path not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext and ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    if not queries or any(not str(q).strip() for q in queries):
        raise HTTPException(status_code=400, detail="queries must be a non-empty list of non-empty strings")

    ckpt = model_names[anatomical_site]
    device = get_device()

    # Load model fresh every request (no caching)
    model, _, preprocess = open_clip.create_model_and_transforms(
        DEFAULT_MODEL_ARCH,
        pretrained=ckpt,
    )
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(DEFAULT_MODEL_ARCH)

    try:
        img = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]
        text_tokens = tokenizer(queries).to(device)              # [N, ctx_len]

        # Autocast only on CUDA
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                img_feat = model.encode_image(image_tensor)
                txt_feat = model.encode_text(text_tokens)
        else:
            img_feat = model.encode_image(image_tensor)
            txt_feat = model.encode_text(text_tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ txt_feat.T).squeeze(0)           # [N]
        probs = torch.softmax(100.0 * sims, dim=-1)         # [N]

        probs_list = probs.detach().float().cpu().tolist()  # python floats
        return list(zip(queries, [float(p) for p in probs_list]))

    finally:
        # Release memory after request
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


@app.post("/match")
def match(req: MatchRequest):
    scored = run_clip_from_path(req.image_path, req.queries, req.anatomical_site)
    scored.sort(key=lambda x: x[1], reverse=True)

    k_eff = min(req.k, len(scored))
    return {
        "image_path": req.image_path,
        "anatomical_site": req.anatomical_site,
        "k": k_eff,
        "results": scored[:k_eff],  # list of (query, probability)
    }
