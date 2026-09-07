import json
from types import SimpleNamespace

import pytest

from pathrag.agents import langgraph_app, tools
from pathrag.vlm.medgemma_client import MedGemmaClient


def _patches():
    return [
        {"id": "P2", "bbox": (0, 0, 10, 10), "score": 2.0},
        {"id": "P0", "bbox": (10, 10, 20, 20), "score": 1.0},
    ]


def test_medgemma_graph_batches_requests_and_preserves_patch_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATHRAG_STAGE4_BACKEND", "medgemma")
    observed = {}

    def fake_crops(image, boxes, out_dir):
        observed["image"] = image
        observed["boxes"] = list(boxes)
        observed["out_dir"] = out_dir
        return [f"{out_dir}/crop-{index}.png" for index in range(2)]

    def fake_batch(self, requests, request_jsonl, answers_jsonl):
        observed["requests"] = requests
        observed["calls"] = observed.get("calls", 0) + 1
        return list(reversed([
            {"request_id": row["request_id"], "patch_id": row["patch_id"], "task_type": row["task_type"], "text": f"{row['task_type']}:{row['patch_id']}", "model_id": "google/medgemma-1.5-4b-it", "precision": "bf16", "quantization": "4bit"}
            for row in requests
        ]))

    monkeypatch.setattr(tools, "save_crops", fake_crops)
    monkeypatch.setattr(MedGemmaClient, "ask_structured_batch", fake_batch)
    state = {"image_path": "slide.png", "question": "Question", "patches": _patches(), "full_captions": ["caption"]}
    result = langgraph_app.n_roi_and_patch_agents(state)
    assert observed["calls"] == 1
    assert len(observed["boxes"]) == 2
    assert len(observed["requests"]) == 4
    assert [row["patch_id"] for row in observed["requests"]] == ["P2", "P2", "P0", "P0"]
    assert result["roi_desc"] == ["roi:P2", "roi:P0"]
    assert result["patch_summaries"] == ["contribution:P2", "contribution:P0"]
    assert result["stage4_provenance"]["request_count"] == 4
    assert result["stage4_provenance"]["model_id"] == "google/medgemma-1.5-4b-it"
    assert result["stage4_provenance"]["precision"] == "bf16"
    assert result["stage4_provenance"]["quantization"] == "4bit"
    assert "caption" in observed["requests"][1]["prompt"]
    assert "caption" not in observed["requests"][0]["prompt"]
    second = langgraph_app.n_roi_and_patch_agents({"image_path": "slide.png", "question": "Question", "patches": _patches(), "full_captions": []})
    assert second["stage4_provenance"]["artifact_dir"] != result["stage4_provenance"]["artifact_dir"]


@pytest.mark.parametrize("kind", ["duplicate", "missing", "unknown", "empty", "malformed"])
def test_structured_answers_reject_invalid_contracts(tmp_path, kind):
    requests = [{"request_id": "r1", "patch_id": "P0", "task_type": "roi"}]
    answer = {"request_id": "r1", "patch_id": "P0", "task_type": "roi", "text": "answer", "model_id": "model", "precision": "bf16", "quantization": "4bit"}
    path = tmp_path / "answers.jsonl"
    if kind == "duplicate":
        path.write_text(json.dumps(answer) + "\n" + json.dumps(answer) + "\n")
    elif kind == "missing":
        path.write_text("")
    elif kind == "unknown":
        answer["request_id"] = "other"; path.write_text(json.dumps(answer) + "\n")
    elif kind == "empty":
        answer["text"] = " "; path.write_text(json.dumps(answer) + "\n")
    else:
        path.write_text("not-json\n")
    with pytest.raises(RuntimeError):
        MedGemmaClient._validated_answers(requests, str(path))


def test_structured_answers_reject_inconsistent_provenance(tmp_path):
    requests = [{"request_id": "r1", "patch_id": "P0", "task_type": "roi"}, {"request_id": "r2", "patch_id": "P0", "task_type": "contribution"}]
    answers = [
        {"request_id": "r1", "patch_id": "P0", "task_type": "roi", "text": "roi", "model_id": "model", "precision": "bf16", "quantization": "4bit"},
        {"request_id": "r2", "patch_id": "P0", "task_type": "contribution", "text": "patch", "model_id": "model", "precision": "fp16", "quantization": "4bit"},
    ]
    path = tmp_path / "answers.jsonl"
    path.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
    with pytest.raises(RuntimeError, match="inconsistent"):
        MedGemmaClient._validated_answers(requests, str(path))


def test_medgemma_timeout_and_oom_diagnostics_are_specific(monkeypatch, tmp_path):
    client = MedGemmaClient()
    monkeypatch.setattr(client, "python", tmp_path / "python")
    monkeypatch.setattr(client, "tool_dir", tmp_path)
    monkeypatch.setattr(client, "config", tmp_path / "config.yaml")
    (tmp_path / "python").write_text("")
    (tmp_path / "config.yaml").write_text("")
    def timeout(*_args, **_kwargs):
        raise tools.subprocess.TimeoutExpired("medgemma", 1)
    monkeypatch.setattr("pathrag.vlm.medgemma_client.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        client._run("requests", ".", "answers")
    monkeypatch.setattr("pathrag.vlm.medgemma_client.subprocess.run", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="CUDA out of memory"))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        client._run("requests", ".", "answers")
