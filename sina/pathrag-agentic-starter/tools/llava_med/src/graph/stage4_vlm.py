import argparse, json, os, sys
from pathlib import Path
import yaml
from rich import print

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--k", type=int, default=None, help="override k_patches")
    ap.add_argument("--critique", type=int, default=None, help="override critique_iters")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.k is not None:
        cfg["stage4"]["k_patches"] = int(args.k)
    if args.critique is not None:
        cfg["stage4"]["critique_iters"] = int(args.critique)

    out_dir = Path(cfg["stage4"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_path = Path(cfg["stage4"]["smoke_answers_path"])
    if not smoke_path.exists() or smoke_path.stat().st_size == 0:
        print("[bold red]Smoke answers file missing or empty.[/bold red]")
        print(f"Expected non-empty at: {smoke_path}")
        sys.exit(2)

    # Stub: in real run, you would call LLaVA‑Med for full image + K patches and collect answers/captions.
    # For now, we just read smoke.jsonl and emit a last_run.json bundle to standardize artifacts.
    records = []
    with open(smoke_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # be permissive: store raw line
                records.append({"raw": line})

    last_run = {
        "k_patches": cfg["stage4"]["k_patches"],
        "critique_iters": cfg["stage4"]["critique_iters"],
        "he_nuclei_threshold": cfg["stage4"]["he_nuclei_threshold"],
        "records_seen": len(records),
        "vlm_repo": cfg["models"]["vlm_repo"],
        "re_rank_mode": cfg["re_rank"]["mode"],
    }
    last_run_path = Path(cfg["stage4"]["last_run_path"])
    with open(last_run_path, "w") as f:
        json.dump(last_run, f, indent=2)

    print("[bold green]Stage‑4 stub complete[/bold green] → wrote", str(last_run_path))
    print(last_run)

if __name__ == "__main__":
    main()
