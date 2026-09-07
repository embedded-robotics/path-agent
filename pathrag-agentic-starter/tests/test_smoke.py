def test_pathrag_is_importable():
    import pathrag

    assert pathrag.__name__ == "pathrag"


def test_build_graph_can_be_constructed_without_running_pipeline():
    from pathrag.agents.langgraph_app import build_graph

    graph = build_graph()

    assert graph is not None
    assert callable(graph.invoke)
