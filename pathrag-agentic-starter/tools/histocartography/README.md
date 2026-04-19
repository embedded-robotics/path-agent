Local Histocartography tool wrapper.

Purpose:
- Internalize the business logic from `Complete_Patch_Extraction_API` without keeping
  the FastAPI service in the runtime path.
- Produce HC-selected patches and non-selected patches using the same 3x3 nuclei-density logic.

Expected runtime:
- A Python environment with:
  - `histocartography`
  - `torch`
  - `openslide-python`
  - `numpy`
  - `Pillow`

Environment variables:
- `HISTOCARTOGRAPHY_PYTHON`: interpreter used to run this tool.
- `HISTOCARTOGRAPHY_CONFIG`: optional config override.

Current bootstrap note:
- For now, the existing `Complete_Patch_Extraction_API/.venv390` can be used as the runtime.
- The long-term target is a dedicated `.venv` under `tools/histocartography`.
