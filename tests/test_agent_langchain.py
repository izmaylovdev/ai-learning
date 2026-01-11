from agents.code_analysis.agent import CodebaseAgent


def test_answer_fallback_uses_stub_llm(tmp_path):
    # create a small file to index
    p = tmp_path / "file.txt"
    p.write_text("This repository contains a model server started in model_server/main.py")

    agent = CodebaseAgent(root_path=str(tmp_path))
    agent.build_index()
    out = agent.answer_question("Where is the model server started?", top_k=1)
    assert out.startswith("[stub LLM]")
    assert "model_server/main.py" in out or "model_server" in out

