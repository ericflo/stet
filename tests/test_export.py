import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_the_servers_copy_of_the_season_is_current():
    from export_season import export

    committed = json.loads((ROOT / "server" / "season.json").read_text())
    assert committed == json.loads(json.dumps(export())), "run: python tools/export_season.py > server/season.json"


def test_the_servers_test_copy_of_season_one_is_current():
    from export_season import export

    committed = json.loads((ROOT / "server" / "testdata" / "season-s1.json").read_text())
    assert committed == json.loads(json.dumps(export("s1"))), "run: python tools/export_season.py s1 > server/testdata/season-s1.json"
