# src/agent/build_graph.py
from langgraph.graph import StateGraph, END
from src.pathrag.agents.nodes.medgemma_reasoner import medgemma_reasoner
# import your other nodes: stage1, stage2, stage3, stage5, ...

def build_graph():
    g = StateGraph(dict)
    g.add_node("stage1", stage1)
    g.add_node("stage2", stage2)
    g.add_node("stage3", stage3)
    g.add_node("stage4", medgemma_reasoner)   # ← MedGemma node
    g.add_node("stage5", stage5)

    g.set_entry_point("stage1")
    g.add_edge("stage1", "stage2")
    g.add_edge("stage2", "stage3")
    g.add_edge("stage3", "stage4")            # ← place after your patch selection
    g.add_edge("stage4", "stage5")

    return g.compile()
