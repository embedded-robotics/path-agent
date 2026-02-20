from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List
from PIL import Image

def save_crops(image_path: str, boxes: Iterable[Tuple[int,int,int,int]], out_dir: str) -> List[str]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    paths: List[str] = []
    for i, (x0,y0,x1,y1) in enumerate(boxes, 1):
        p = out / f"{Path(image_path).stem}_P{i}.png"
        img.crop((x0,y0,x1,y1)).save(p)
        paths.append(str(p))
    return paths
