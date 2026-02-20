from pathrag.agents.tools import run_histocartography, make_llava_query_files, PatchInfo
if __name__ == "__main__":
    image_path = "sample_he.png"; question = "What are a few well-developed cell nests with?"
    info = run_histocartography(image_path, top_k=3)
    patches = [PatchInfo(**p) for p in info["patches"]]
    files = make_llava_query_files(image_path, question, patches)
    print("Wrote query JSONLs:", files)