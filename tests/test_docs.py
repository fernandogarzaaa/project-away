from away import config as C
from away.bindings import Action


def test_readme_example_config_is_valid():
    from pathlib import Path

    import yaml

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = readme.split("### Example `config.yaml`")[1].split("```yaml")[1].split("```")[0]
    cfg, unknown = C.from_dict(yaml.safe_load(block))
    assert unknown == []
    b = cfg.bindings
    assert b.profile_for_app("Code.exe") == "code"
    assert b.resolve("gestures", "open_palm", "code") == Action("key", "ctrl+shift+p")
    assert b.resolve("voice", "Close tab.", "chrome") == Action("key", "ctrl+w")
