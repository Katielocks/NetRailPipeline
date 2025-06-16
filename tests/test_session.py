import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rail_data.rail_io.session import Session, CredentialsError

class DummyRequestSession:
    def __init__(self):
        self.headers = {}
        self.auth = None
    def request(self, method, url, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"ok": True}
        return Resp()
    def mount(self, prefix, adapter):
        pass


def test_session_requires_credentials():
    with pytest.raises(CredentialsError):
        Session()


def test_session_get_json():
    s = Session(token="abc", session=DummyRequestSession(), retries=0)
    result = s.get_json("http://example.com")
    assert result == {"ok": True}
    assert s._s.headers[Session._TOKEN_HEADER] == "abc"