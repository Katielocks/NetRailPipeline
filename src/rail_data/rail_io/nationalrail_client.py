from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import Any, Dict, Iterable, Mapping, Optional
import os
from session import Session

__all__ = ["NationalRailSession", "STATIC_FEEDS"]
__version__ = "0.2.0"

_STAT_FEEDS = "https://opendata.nationalrail.co.uk/api/staticfeeds"

STATIC_FEEDS: Mapping[str, str] = {
    "fares": f"{_STAT_FEEDS}/2.0/fares",
    "routeing": f"{_STAT_FEEDS}/2.0/routeing",
    "timetable": f"{_STAT_FEEDS}/3.0/timetable",
    "service_indicators": f"{_STAT_FEEDS}/4.0/serviceIndicators",
    "tocs": f"{_STAT_FEEDS}/4.0/tocs",
    "ticket_restrictions": f"{_STAT_FEEDS}/4.0/ticket-restrictions",
    "ticket_types": f"{_STAT_FEEDS}/4.0/ticket-types",
    "public_promotions": f"{_STAT_FEEDS}/4.0/promotions-public",
    "stations": f"{_STAT_FEEDS}/4.0/stations",
    "incidents": f"{_STAT_FEEDS}/5.0/incidents",
}

_HSP_ROOT = "https://hsp-prod.rockshore.net/api/v1"
_SERVICE_DETAILS = f"{_HSP_ROOT}/serviceDetails"
_SERVICE_METRICS = f"{_HSP_ROOT}/serviceMetrics"


class NationalRailSession(Session):
    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.user: Optional[str] = user or os.getenv("NATR_USER")
        self.password: Optional[str] = password or os.getenv("NATR_PASS")
        self.token: Optional[str] = token or os.getenv("NATR_TOKEN")
        super().__init__(self.user, self.password, self.token)

    def _static(self, key: str, *, save_to: str | Path | None = None) -> str:
        url = STATIC_FEEDS[key]
        if save_to:
            self.save(url, save_to)
            return str(Path(save_to).resolve())
        return self.download_text(url)

    def fares(self, *, save_to: str | Path | None = None) -> str:
        return self._static("fares", save_to=save_to)

    def routeing(self, *, save_to: str | Path | None = None) -> str:
        return self._static("routeing", save_to=save_to)

    def timetable(self, *, save_to: str | Path | None = None) -> str:
        return self._static("timetable", save_to=save_to)

    def service_indicators(self, *, save_to: str | Path | None = None) -> str:
        return self._static("service_indicators", save_to=save_to)

    def tocs(self, *, save_to: str | Path | None = None) -> str:
        return self._static("tocs", save_to=save_to)

    def ticket_restrictions(self, *, save_to: str | Path | None = None) -> str:
        return self._static("ticket_restrictions", save_to=save_to)

    def ticket_types(self, *, save_to: str | Path | None = None) -> str:
        return self._static("ticket_types", save_to=save_to)

    def public_promotions(self, *, save_to: str | Path | None = None) -> str:
        return self._static("public_promotions", save_to=save_to)

    def stations(self, *, save_to: str | Path | None = None) -> str:
        return self._static("stations", save_to=save_to)

    def incidents(self, *, save_to: str | Path | None = None) -> str:
        return self._static("incidents", save_to=save_to)

    def service_details(self, rid: str) -> Dict[str, Any]:
        payload = {"rid": rid}
        resp = self.post(
            _SERVICE_DETAILS,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    def service_metrics(
        self,
        *,
        from_loc: str,
        to_loc: str,
        from_time: str,
        to_time: str,
        from_date: date | str,
        to_date: date | str,
        days: str = "WEEKDAY",
        toc_filter: Iterable[str] | None = None,
        tolerance: Iterable[int] | None = None,
    ) -> Dict[str, Any]:
        def _format_date(x: date | str) -> str:
            return x if isinstance(x, str) else x.isoformat()

        payload: Dict[str, Any] = {
            "from_loc": from_loc.upper(),
            "to_loc": to_loc.upper(),
            "from_time": from_time,
            "to_time": to_time,
            "from_date": _format_date(from_date),
            "to_date": _format_date(to_date),
            "days": days.upper(),
        }

        if toc_filter is not None:
            payload["toc_filter"] = [code.upper() for code in toc_filter]
        if tolerance is not None:
            payload["tolerance"] = [str(mins) for mins in tolerance]

        resp = self.post(
            _SERVICE_METRICS,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def download(
    url: str,
    dest: str | Path | None = None,
    *,
    token: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> str:
    with NationalRailSession(user=user, password=password, token=token) as ses:
        if dest:
            ses.save(url, dest)
            return str(Path(dest).resolve())
        return ses.download_text(url)
