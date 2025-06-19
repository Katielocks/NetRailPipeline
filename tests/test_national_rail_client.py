import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.io import national_rail_client as nr
from rail_data.io.session import Session


class DummyRequestSession:
    def __init__(self):
        self.headers = {}
    def request(self, method, url, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"ok": True}
            @property
            def text(self):
                return "ok"
            @property
            def content(self):
                return b"ok"
        return Resp()
    def mount(self, prefix, adapter):
        pass
    def close(self):
        pass

class DummyNRS(nr.NationalRailSession):
    def __init__(self, *args, **kwargs):
        # bypass NationalRailSession.__init__ which expects credentials
        Session.__init__(self, token="t", session=DummyRequestSession())
    def download_text(self, url, **kwargs):
        self.last_url = url
        return "text:" + url
    def save(self, url, dest, **kwargs):
        Path(dest).write_text("saved")
        self.saved = (url, dest)
        return Path(dest)


def test_static_download():
    s = DummyNRS()
    out = s.fares()
    assert out == "text:" + nr.STATIC_FEEDS["fares"]
    assert s.last_url == nr.STATIC_FEEDS["fares"]


def test_static_save(tmp_path):
    s = DummyNRS()
    dest = tmp_path / "f.json"
    out = s.stations(save_to=dest)
    assert dest.read_text() == "saved"
    assert out == str(dest.resolve())
    assert s.saved[0] == nr.STATIC_FEEDS["stations"]


def test_download_helper(monkeypatch):
    monkeypatch.setattr(nr, "NationalRailSession", DummyNRS)
    out = nr.download("http://x")
    assert out == "text:http://x"