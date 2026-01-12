from agents.code_analysis.agent import analyze_code
from agents.code_analysis.tools import initialize_code_analysis


def test_analyze_code_with_tools(tmp_path):
    """Test the analyze_code tool can answer questions about the repository."""
    # Create a small file to index
    p = tmp_path / "file.txt"
    p.write_text("This repository contains a model server started in model_server/main.py")

    # Initialize code analysis with the test directory
    initialize_code_analysis(str(tmp_path))

    # Use the analyze_code tool
    result = analyze_code.invoke({"question": "Where is the model server started?"})

    # The result should contain information about the file or model_server
    assert isinstance(result, str)
    assert len(result) > 0


if __name__ == "__main__":
    test_analyze_code_with_tools()