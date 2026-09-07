import json
import math
import os
from types import SimpleNamespace

import pytest

from pathrag import local_dev
from pathrag.site_labeler import predict_tissue


def canonical_patches():
    return [
        {"id": "P0", "bbox": [0, 0, 10, 10], "score": 2},
        {"id": "P1", "bbox": [10, 10, 20, 20], "score": 1.5},
    ]


@pytest.mark.parametrize(
    "document",
    [
        canonical_patches(),
        {"patches": canonical_patches()},
        {"fuse": {"patches": canonical_patches()}},
    ],
)
def test_canonical_saved_patch_formats(document):
    assert local_dev.normalize_patch_document(document) == [
        {"id": "P0", "bbox": [0, 0, 10, 10], "score": 2.0},
        {"id": "P1", "bbox": [10, 10, 20, 20], "score": 1.5},
    ]


def test_chief_coordinate_wrapper_is_normalized():
    document = {
        "chief_patch_coords": [
            {"x1": 0, "y1": 1, "x2": 10, "y2": 11},
            {"x1": 20, "y1": 21, "x2": 30, "y2": 31},
        ]
    }

    assert local_dev.normalize_patch_document(document) == [
        {"id": "CP0", "bbox": [0, 1, 10, 11], "score": 2.0},
        {"id": "CP1", "bbox": [20, 21, 30, 31], "score": 1.0},
    ]


def test_malformed_json_reports_location(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"patches": [', encoding="utf-8")

    with pytest.raises(local_dev.LocalDevError, match=r"line 1, column"):
        local_dev.load_patches_json(source)


def test_empty_patch_list_is_rejected():
    with pytest.raises(local_dev.LocalDevError, match="at least one patch"):
        local_dev.normalize_patch_document([])


def test_full_evaluation_sheet_is_rejected_actionably():
    with pytest.raises(local_dev.LocalDevError, match="extract one question"):
        local_dev.normalize_patch_document({"questions": []})


@pytest.mark.parametrize(
    ("patches", "message"),
    [
        (
            [
                {"id": "P0", "bbox": [0, 0, 10, 10], "score": 1},
                {"id": "P0", "bbox": [10, 10, 20, 20], "score": 1},
            ],
            "duplicates patch ID",
        ),
        ([{"id": "", "bbox": [0, 0, 10, 10], "score": 1}], "non-empty string"),
        ([{"id": "P0", "bbox": [0, 0, 10], "score": 1}], "exactly four"),
        ([{"id": "P0", "bbox": [0, 0, 1.5, 10], "score": 1}], "finite integer"),
        ([{"id": "P0", "bbox": [0, 0, math.inf, 10], "score": 1}], "finite integer"),
        ([{"id": "P0", "bbox": [0, 0, True, 10], "score": 1}], "finite integer"),
        ([{"id": "P0", "bbox": [10, 0, 10, 10], "score": 1}], "x2 > x1"),
        ([{"id": "P0", "bbox": [0, 10, 10, 5], "score": 1}], "y2 > y1"),
        ([{"id": "P0", "bbox": [0, 0, 10, 10], "score": "1"}], "finite number"),
        ([{"id": "P0", "bbox": [0, 0, 10, 10], "score": math.nan}], "finite number"),
        ([{"id": "P0", "bbox": [0, 0, 10, 10], "score": True}], "finite number"),
    ],
)
def test_invalid_patch_fields_are_rejected(patches, message):
    with pytest.raises(local_dev.LocalDevError, match=message):
        local_dev.normalize_patch_document(patches)


def test_top_k_selection_is_ordered_and_clamps():
    patches = canonical_patches()

    selected, message = local_dev.select_patches(patches, 1)
    clamped, clamp_message = local_dev.select_patches(patches, 5)

    assert [patch["id"] for patch in selected] == ["P0"]
    assert message is None
    assert [patch["id"] for patch in clamped] == ["P0", "P1"]
    assert "using all 2 patches" in clamp_message


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(local_dev.LocalDevError, match="top_k"):
        local_dev.select_patches(canonical_patches(), top_k)


@pytest.mark.parametrize(
    ("image_path", "question", "mode", "max_rounds", "message"),
    [
        ("logical.png", "", "answer", 1, "question"),
        ("logical.png", "question", "invalid", 1, "mode"),
        ("logical.png", "question", "answer", 0, "always executes one critique"),
        ("logical.png", "question", "answer", -1, "max_rounds"),
    ],
)
def test_invalid_run_options_are_rejected(image_path, question, mode, max_rounds, message):
    with pytest.raises(local_dev.LocalDevError, match=message):
        local_dev.validate_run_options(image_path, question, mode, max_rounds)


@pytest.mark.parametrize("key_value", [None, "", "   "], ids=["missing", "empty", "whitespace"])
def test_missing_or_blank_openai_key_fails_before_production_graph_import(
    tmp_path, monkeypatch, key_value
):
    source = tmp_path / "patches.json"
    source.write_text(json.dumps(canonical_patches()), encoding="utf-8")
    if key_value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key_value)
    dotenv_calls = []
    monkeypatch.setattr(local_dev, "load_dotenv", lambda: dotenv_calls.append(True))

    def forbidden_import():
        pytest.fail("production graph was imported before the OPENAI key check")

    monkeypatch.setattr(local_dev, "_load_production_graph", forbidden_import)
    with pytest.warns(UserWarning):
        with pytest.raises(local_dev.LocalDevError, match="OPENAI_API_KEY"):
            local_dev.run_local_dev(
                patches_json=source,
                image_path="missing.png",
                question="Question",
                out=tmp_path / "out.json",
            )
    assert dotenv_calls == [True]


@pytest.mark.parametrize("key_value", [None, "", "   "], ids=["missing", "empty", "whitespace"])
def test_missing_or_blank_openai_key_fails_before_injected_factory(
    tmp_path, monkeypatch, key_value
):
    source = tmp_path / "patches.json"
    source.write_text(json.dumps(canonical_patches()), encoding="utf-8")
    if key_value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key_value)
    monkeypatch.setattr(local_dev, "load_dotenv", lambda: None)

    def forbidden_factory():
        pytest.fail("injected graph factory was called before the OPENAI key check")

    with pytest.warns(UserWarning):
        with pytest.raises(local_dev.LocalDevError, match="OPENAI_API_KEY"):
            local_dev.run_local_dev(
                patches_json=source,
                image_path="missing.png",
                question="Question",
                out=tmp_path / "out.json",
                graph_factory=forbidden_factory,
            )


def test_production_graph_is_loaded_after_successful_key_preflight(
    tmp_path, monkeypatch
):
    source = tmp_path / "patches.json"
    source.write_text(json.dumps(canonical_patches()), encoding="utf-8")
    events = []

    def fake_load_dotenv():
        events.append("dotenv")
        monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    class FakeGraph:
        def invoke(self, state, config):
            events.append("invoke")
            return {
                **state,
                "subpath_label": "test-label",
                "full_captions": ["test caption"],
                "roi_useful": [True],
                "roi_desc": ["[stub-roi:P0] test"],
                "patch_summaries": ["test summary"],
                "chosen_idx": [0],
                "final_answer": "test answer",
                "round_ix": 1,
                "dynamic_k": 1,
            }

    def fake_production_loader():
        assert events == ["dotenv"]
        assert os.environ["OPENAI_API_KEY"] == "test-only-key"
        events.append("production-loader")
        return FakeGraph()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(local_dev, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(local_dev, "_load_production_graph", fake_production_loader)

    with pytest.warns(UserWarning):
        result = local_dev.run_local_dev(
            patches_json=source,
            image_path="missing.png",
            question="Question",
            top_k=1,
            out=tmp_path / "out.json",
        )

    assert events == ["dotenv", "production-loader", "invoke"]
    assert result["run"]["requested_top_k"] == 1
    assert result["run"]["effective_top_k"] == 1
    assert result["run"]["top_k"] == 1


def test_production_loader_reuses_module_level_app_without_second_construction(
    monkeypatch,
):
    app = object()
    imports = []

    def forbidden_build_graph():
        pytest.fail("local-dev constructed a second production graph")

    module = SimpleNamespace(app=app, build_graph=forbidden_build_graph)

    def fake_import_module(name):
        imports.append(name)
        return module

    monkeypatch.setattr(local_dev.importlib, "import_module", fake_import_module)

    assert local_dev._load_production_graph() is app
    assert imports == ["pathrag.agents.langgraph_app"]


def test_output_must_not_overwrite_patch_input(tmp_path):
    source = tmp_path / "patches.json"
    source.write_text(json.dumps(canonical_patches()), encoding="utf-8")

    with pytest.raises(local_dev.LocalDevError, match="must not overwrite"):
        local_dev.run_local_dev(
            patches_json=source,
            image_path="logical.png",
            question="Question",
            out=source,
        )


def test_stage3_placeholder_uses_path_string_and_packaged_bank(tmp_path, monkeypatch):
    from pathrag.agents import tools

    image = tmp_path / "image.png"
    image.write_text("first contents", encoding="utf-8")
    first = predict_tissue(str(image))
    image.write_text("different contents", encoding="utf-8")
    second = predict_tissue(str(image))
    image.unlink()
    missing = predict_tissue(str(image))

    assert first == second == missing

    monkeypatch.setenv("PATHRAG_USE_RETRIEVER_API", "0")
    label = tools.identify_subpathology(str(image))

    def forbidden_request(*_args, **_kwargs):
        pytest.fail("legacy localhost PathGenCLIP retriever was called")

    monkeypatch.setattr(tools, "_post_json", forbidden_request)
    captions = tools.retrieve_subpath_captions(label, top_m=2)
    assert captions
    assert all(isinstance(caption, str) for caption in captions)


def test_placeholder_warning_is_always_emitted(tmp_path):
    image = tmp_path / "exists.png"
    image.write_text("not inspected", encoding="utf-8")

    with pytest.warns(UserWarning) as existing_warnings:
        local_dev._emit_stage3_warnings(str(image))
    with pytest.warns(UserWarning) as missing_warnings:
        local_dev._emit_stage3_warnings(str(tmp_path / "missing.png"))

    assert len(existing_warnings) == 1
    assert "placeholder-only" in str(existing_warnings[0].message)
    assert len(missing_warnings) == 2
    assert "does not exist" in str(missing_warnings[1].message)


def test_offline_runner_integration_skips_heavy_stages_and_restores_environment(
    tmp_path, monkeypatch, capsys
):
    from pathrag.agents import langgraph_app, tools

    source = tmp_path / "patches.json"
    source.write_text(json.dumps({"patches": canonical_patches()}), encoding="utf-8")
    output = tmp_path / "nested" / "result.json"
    missing_image = str(tmp_path / "missing.png")

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("PATHRAG_STAGE4_BACKEND", "medgemma")
    monkeypatch.setenv("PATHRAG_STAGE4_REMOTE_URL", "http://localhost:9999")
    monkeypatch.setenv("PATHRAG_USE_RETRIEVER_API", "1")

    def forbidden(*_args, **_kwargs):
        pytest.fail("offline local-dev integration invoked a forbidden operation")

    for name in ("tile_image", "histocartography_rank", "common_patches"):
        monkeypatch.setattr(langgraph_app, name, forbidden)
    for name in (
        "HistocartographyClient",
        "ChiefClient",
        "MedGemmaClient",
        "LlavaMedClient",
        "save_crops",
        "_write_jsonl",
        "_call_stage4_remote",
        "_get_stage4_client",
        "_post_json",
        "_get_openai_client",
    ):
        monkeypatch.setattr(tools, name, forbidden)
    monkeypatch.setattr(tools.subprocess, "run", forbidden)
    monkeypatch.setattr(local_dev, "_load_production_graph", forbidden)

    observed = {}

    class FakeGraph:
        def invoke(self, state, config):
            observed["backend"] = os.environ.get("PATHRAG_STAGE4_BACKEND")
            observed["remote"] = os.environ.get("PATHRAG_STAGE4_REMOTE_URL")
            observed["retriever"] = os.environ.get("PATHRAG_USE_RETRIEVER_API")
            observed["state"] = dict(state)
            observed["config"] = config

            state = langgraph_app.n_tile_and_rank(state)
            state = langgraph_app.n_identify_and_retrieve(state)
            state = langgraph_app.n_roi_and_patch_agents(state)
            state = langgraph_app.n_rerank_and_choose(state)
            state["round_ix"] = 1
            state["final_answer"] = "offline fake final answer"
            return state

    with pytest.warns(UserWarning) as run_warnings:
        result = local_dev.run_local_dev(
            patches_json=source,
            image_path=missing_image,
            question="  Alignment   question  ",
            top_k=5,
            max_rounds=1,
            mode="answer",
            out=output,
            thread_id="explicit-thread",
            graph_factory=lambda: FakeGraph(),
        )

    assert len(run_warnings) == 3
    assert observed["backend"] == "stub"
    assert observed["remote"] is None
    assert observed["retriever"] == "0"
    assert observed["state"]["top_k"] == 2
    assert [patch["id"] for patch in observed["state"]["patches"]] == ["P0", "P1"]
    assert observed["config"] == {"configurable": {"thread_id": "explicit-thread"}}
    assert os.environ["PATHRAG_STAGE4_BACKEND"] == "medgemma"
    assert os.environ["PATHRAG_STAGE4_REMOTE_URL"] == "http://localhost:9999"
    assert os.environ["PATHRAG_USE_RETRIEVER_API"] == "1"

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == result
    assert result["run"]["backend"] == "stub"
    assert result["run"]["top_k"] == 2
    assert result["run"]["requested_top_k"] == 5
    assert result["run"]["effective_top_k"] == 2
    assert result["run"]["thread_id"] == "explicit-thread"
    assert result["question"] == "Alignment   question"
    assert [patch["id"] for patch in result["patches"]] == ["P0", "P1"]
    assert len(result["roi_useful"]) == 2
    assert len(result["roi_desc"]) == 2
    assert len(result["patch_summaries"]) == 2
    assert "subpath_label" in result
    assert result["full_captions"]
    assert "chosen_idx" in result
    assert result["final_answer"] == "offline fake final answer"
    assert result["round_ix"] == 1
    assert result["dynamic_k"] == len(result["chosen_idx"])
    assert output.is_file()

    console = capsys.readouterr().out
    assert str(output.resolve()) in console
    assert "['P0', 'P1']" in console
    assert "Chosen indices:" in console
    assert "offline fake final answer" in console


def test_stage4_backend_parser_defaults_to_stub_and_accepts_medgemma():
    parser = local_dev.build_parser()
    required = ["--patches-json", "patches.json", "--image-path", "image.png", "--question", "Question"]

    assert parser.parse_args(required).stage4_backend == "stub"
    assert parser.parse_args([*required, "--stage4-backend", "stub"]).stage4_backend == "stub"
    assert parser.parse_args([*required, "--stage4-backend", "medgemma"]).stage4_backend == "medgemma"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--stage4-backend", "llava-med"])


def test_medgemma_mode_reaches_graph_and_preserves_medgemma_environment(tmp_path, monkeypatch):
    source = tmp_path / "patches.json"
    source.write_text(json.dumps(canonical_patches()), encoding="utf-8")
    expected = {
        "MEDGEMMA_MODEL": "authorized-model",
        "MEDGEMMA_PRECISION": "fp16",
        "MEDGEMMA_QUANTIZATION": "8bit",
        "MEDGEMMA_MAX_NEW_TOKENS": "123",
        "MEDGEMMA_SUBPROCESS_TIMEOUT_SECONDS": "456",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/cache-root",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("PATHRAG_STAGE4_REMOTE_URL", "http://llava.example")
    monkeypatch.setenv("PATHRAG_USE_RETRIEVER_API", "1")
    monkeypatch.setattr(local_dev, "load_dotenv", lambda: None)
    observed = {}

    class FakeGraph:
        def invoke(self, state, config):
            observed["backend"] = os.environ.get("PATHRAG_STAGE4_BACKEND")
            observed["remote"] = os.environ.get("PATHRAG_STAGE4_REMOTE_URL")
            observed["retriever"] = os.environ.get("PATHRAG_USE_RETRIEVER_API")
            observed["medgemma"] = {name: os.environ.get(name) for name in expected}
            return {**state, "roi_useful": [True, True], "roi_desc": ["roi-0", "roi-1"], "patch_summaries": ["patch-0", "patch-1"], "chosen_idx": [0], "final_answer": "answer"}

    with pytest.warns(UserWarning):
        result = local_dev.run_local_dev(
            patches_json=source,
            image_path="missing.png",
            question="Question",
            stage4_backend="medgemma",
            out=tmp_path / "result.json",
            graph_factory=FakeGraph,
        )

    assert observed == {"backend": "medgemma", "remote": None, "retriever": "0", "medgemma": expected}
    assert result["run"]["backend"] == "medgemma"
    assert os.environ["PATHRAG_STAGE4_REMOTE_URL"] == "http://llava.example"
    assert os.environ["PATHRAG_USE_RETRIEVER_API"] == "1"
    assert {name: os.environ[name] for name in expected} == expected


def test_medgemma_environment_is_restored_when_graph_raises(tmp_path, monkeypatch):
    source = tmp_path / "patches.json"
    source.write_text(json.dumps(canonical_patches()), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("PATHRAG_STAGE4_BACKEND", "llava-med")
    monkeypatch.setenv("PATHRAG_STAGE4_REMOTE_URL", "http://llava.example")
    monkeypatch.setenv("PATHRAG_USE_RETRIEVER_API", "1")
    monkeypatch.setattr(local_dev, "load_dotenv", lambda: None)

    class FailingGraph:
        def invoke(self, *_args, **_kwargs):
            assert os.environ["PATHRAG_STAGE4_BACKEND"] == "medgemma"
            raise RuntimeError("graph failure")

    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError, match="graph failure"):
            local_dev.run_local_dev(
                patches_json=source,
                image_path="missing.png",
                question="Question",
                stage4_backend="medgemma",
                out=tmp_path / "result.json",
                graph_factory=FailingGraph,
            )

    assert os.environ["PATHRAG_STAGE4_BACKEND"] == "llava-med"
    assert os.environ["PATHRAG_STAGE4_REMOTE_URL"] == "http://llava.example"
    assert os.environ["PATHRAG_USE_RETRIEVER_API"] == "1"


def test_generated_thread_id_has_stable_prefix():
    assert local_dev._thread_id(None).startswith("local-dev-")
