import pytest

from pathrag.agents import tools
from pathrag.agents.langgraph_app import n_roi_and_patch_agents


@pytest.fixture
def guarded_stub_backend(monkeypatch):
    monkeypatch.setenv("PATHRAG_STAGE4_BACKEND", "stub")
    monkeypatch.delenv("PATHRAG_STAGE4_REMOTE_URL", raising=False)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Stub backend invoked a forbidden Stage 4 operation")

    for name in (
        "save_crops",
        "_write_jsonl",
        "_call_stage4_remote",
        "_get_stage4_client",
        "MedGemmaClient",
        "LlavaMedClient",
    ):
        monkeypatch.setattr(tools, name, forbidden)
    monkeypatch.setattr(tools.subprocess, "run", forbidden)


def test_stub_backend_is_accepted(guarded_stub_backend):
    assert tools._get_stage4_backend() == "stub"


def test_stub_roi_is_deterministic_without_an_image(guarded_stub_backend):
    patch = tools.Patch(id="P0", bbox=(0, 0, 10, 10))

    first = tools.roi_agent_describe(patch, "Any question", "/missing/slide.svs")
    second = tools.roi_agent_describe(patch, "Any question", "/missing/slide.svs")

    assert first == second
    assert first == {
        "useful": True,
        "description": "[stub-roi:P0] Deterministic ROI description for pipeline testing.",
    }


def test_stub_patch_output_is_deterministic_and_distinguishes_ids(guarded_stub_backend):
    question = "  Check   patch\norder  "
    patch_zero = tools.Patch(id="P0", bbox=(0, 0, 10, 10))
    patch_one = tools.Patch(id="P1", bbox=(10, 10, 20, 20))

    first = tools.patch_agent_contribution(
        patch_zero, question, [], "/missing/slide.svs"
    )
    repeated = tools.patch_agent_contribution(
        patch_zero, question, [], "/missing/slide.svs"
    )
    different_patch = tools.patch_agent_contribution(
        patch_one, question, [], "/missing/slide.svs"
    )

    assert first == repeated
    assert first == "[stub-patch:P0] Deterministic contribution for question: Check patch order"
    assert different_patch.startswith("[stub-patch:P1]")
    assert different_patch != first


def test_stub_question_text_has_fixed_maximum_length(guarded_stub_backend):
    question = "word " * 100
    result = tools.patch_agent_contribution(
        tools.Patch(id="P0", bbox=(0, 0, 10, 10)),
        question,
        [],
        "/missing/slide.svs",
    )
    question_text = result.split("question: ", 1)[1]

    assert len(question_text) == tools.STUB_QUESTION_MAX_CHARS
    assert question_text.endswith("...")


def test_stage4_node_preserves_stub_count_and_order(guarded_stub_backend):
    state = {
        "image_path": "/missing/slide.svs",
        "question": "Alignment check",
        "patches": [
            {"id": "P2", "bbox": (0, 0, 10, 10), "score": 2.0},
            {"id": "P0", "bbox": (10, 10, 20, 20), "score": 1.0},
        ],
        "full_captions": [],
    }

    result = n_roi_and_patch_agents(state)

    assert result["roi_useful"] == [True, True]
    assert len(result["roi_desc"]) == 2
    assert len(result["patch_summaries"]) == 2
    assert result["roi_desc"][0].startswith("[stub-roi:P2]")
    assert result["roi_desc"][1].startswith("[stub-roi:P0]")
    assert result["patch_summaries"][0].startswith("[stub-patch:P2]")
    assert result["patch_summaries"][1].startswith("[stub-patch:P0]")


def test_invalid_backend_error_lists_supported_values(monkeypatch):
    monkeypatch.setenv("PATHRAG_STAGE4_BACKEND", "unknown")

    with pytest.raises(RuntimeError) as exc_info:
        tools._get_stage4_backend()

    message = str(exc_info.value)
    assert "stub" in message
    assert "medgemma" in message
    assert "llava-med" in message
