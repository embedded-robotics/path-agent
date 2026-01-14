from pathrag.agents.autogen_app import build_agents
if __name__ == "__main__":
    user, planner = build_agents()
    image_path = "sample_he.png"; question = "What are a few well-developed cell nests with?"
    task = f"Run Path-RAG on {image_path} with question '{question}' in 'answer' mode and top_k=3. Return only the final fused answer."
    result = user.initiate_chat(planner, message=task)
    print("FINAL ANSWER:", result.summary)