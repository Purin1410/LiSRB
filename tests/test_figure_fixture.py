import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "figure"
    / "generate_f1_fixture.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_f1_fixture", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f1_fixture_is_deterministic_and_ordered():
    generator = _load_generator()
    first = generator.build_fixture()
    second = generator.build_fixture()

    assert first == second
    assert first["display_pair"]["lcp_path"] == ["ROOT"]
    assert first["display_pair"]["distance"] == 2
    assert first["display_pair"]["ordered_type_pair"] == ["NUM", "DEN"]
    assert first["display_pair"]["r_ij"] != first["display_pair"]["r_ji"]
    assert first["config"]["num_relations"] == 16 * 5 * 5
