from typer.testing import CliRunner
from kb_vectorizer.cli_deperecated import app

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Vectorize a knowledge base" in result.stdout
