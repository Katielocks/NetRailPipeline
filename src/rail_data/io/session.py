from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
import logging

log = logging.getLogger(__name__)

import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CredentialsError(RuntimeError):
    """Raised when neither token nor (user & password) are supplied."""


class Session:

    _TOKEN_HEADER = "X-Auth-Token"

    def __init__(
        self,
        *,
        user: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: float | tuple[float, float] = (10.0, 60.0), 
        retries: int = 3,
        backoff_factor: float = 0.5,
        status_forcelist: Iterable[int] | None = None,
        session: requests.Session | None = None,
    ) -> None:


        self.timeout: float | tuple[float, float] = timeout
        self.user: Optional[str] = user
        self.password: Optional[str] = password
        self.token: Optional[str] = token

        if not self.token and (not self.user or not self.password):
            raise CredentialsError(
                "You must supply either a bearer token or both user & password"
            )

        self._s = session or requests.Session()
        log.debug("Session created with timeout=%s", self.timeout)

        if self.token:
            self._s.headers[self._TOKEN_HEADER] = self.token
        else:
            self._s.auth = HTTPBasicAuth(self.user, self.password) 

        if retries:
            retry_strategy = Retry(
                total=retries,
                connect=retries,
                read=retries,
                status=retries,
                backoff_factor=backoff_factor,
                status_forcelist=tuple(status_forcelist or (500, 502, 503, 504)),
                allowed_methods=frozenset({"HEAD", "GET", "OPTIONS"}),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._s.mount("https://", adapter)
            self._s.mount("http://", adapter)


    def __enter__(self) -> "Session":
        log.debug("Entering session context")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _req(self, method: str, url: str, **kwargs) -> requests.Response:
        log.debug("%s %s", method, url)
        kwargs.setdefault("timeout", self.timeout)
        resp = self._s.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def get(self, url: str, **kwargs) -> requests.Response:
        return self._req("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self._req("POST", url, **kwargs)

    def get_json(self, url: str, **kwargs) -> Any:
    
        resp = self.get(url, **kwargs)
        try:
            return resp.json()
        except ValueError as err:
            raise ValueError(
                f"Expected JSON response from {url!r} "
                f"but could not decode it"
            ) from err

    def post_json(
        self,
        url: str,
        json: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Any:

        kwargs["json"] = json
        resp = self.post(url, **kwargs)
        try:
            return resp.json()
        except ValueError as err:  # pragma: no cover
            raise ValueError(
                f"Expected JSON response from {url!r} "
                f"but could not decode it"
            ) from err

    def download_text(self, url: str, **kwargs) -> str:
        return self.get(url, **kwargs).text

    def download_binary(self, url: str, **kwargs) -> bytes:
        return self.get(url, **kwargs).content

    def save(self, url: str, dest: str | Path, **kwargs) -> Path:
        log.info("Downloading %s to %s", url, dest)   
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(self.download_binary(url, **kwargs))
        tmp.rename(dest)
        return dest

    def close(self) -> None:
        log.debug("Closing session")
        self._s.close()

