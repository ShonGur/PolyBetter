#!/usr/bin/env python3
"""Collect historical odds for 100 closed Sports-Betting events from Polymarket and write Parquet."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class PolymarketAPIError(RuntimeError):
    """Raised when the Polymarket API cannot be queried or parsed."""


@dataclass
class OddsSnapshot:
    timestamp: str
    yes_odds: Optional[float]
    no_odds: Optional[float]


@dataclass
class SportsBettingEvent:
    event_id: str
    title: str
    event_url: str
    final_result: str
    odds_15m: List[OddsSnapshot]

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record["odds_15m"] = json.dumps(record["odds_15m"])
        return record


class PolymarketClient:
    """Minimal user-less client for Polymarket Gamma + CLOB endpoints."""

    gamma_base_url = "https://gamma-api.polymarket.com"
    clob_base_url = "https://clob.polymarket.com"

    def __init__(self, timeout_seconds: int = 30, pause_seconds: float = 0.15) -> None:
        self.timeout_seconds = timeout_seconds
        self.pause_seconds = pause_seconds

    def _get_json(self, base_url: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{base_url}{endpoint}{query}"
        request = Request(url, headers={"User-Agent": "PolyBetter-DataCollector/1.0"})

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except Exception as exc:  # pragma: no cover - depends on network/API status
            raise PolymarketAPIError(f"Failed calling {url}: {exc}") from exc

    def fetch_closed_sports_events(self, desired_count: int = 100, batch_size: int = 200) -> List[Dict[str, Any]]:
        """Fetch closed events and keep only Sports-Betting events."""
        events: List[Dict[str, Any]] = []
        offset = 0

        while len(events) < desired_count:
            payload = self._get_json(
                self.gamma_base_url,
                "/events",
                {
                    "limit": batch_size,
                    "offset": offset,
                    "closed": "true",
                    "active": "false",
                    "archived": "false",
                },
            )
            if not isinstance(payload, list) or not payload:
                break

            for event in payload:
                if self._is_sports_betting_event(event):
                    events.append(event)
                    if len(events) >= desired_count:
                        break

            offset += batch_size
            time.sleep(self.pause_seconds)

        return events[:desired_count]

    @staticmethod
    def _is_sports_betting_event(event: Dict[str, Any]) -> bool:
        """Flexible matcher because event taxonomy fields vary over time."""
        candidate_fields: List[str] = []
        for key in ("category", "type", "eventType", "slug"):
            value = event.get(key)
            if isinstance(value, str):
                candidate_fields.append(value.lower())

        for tag in event.get("tags", []) or []:
            if isinstance(tag, dict):
                for tag_key in ("slug", "label", "name"):
                    tag_value = tag.get(tag_key)
                    if isinstance(tag_value, str):
                        candidate_fields.append(tag_value.lower())

        normalized = " ".join(candidate_fields)
        return "sports-betting" in normalized or "sports" in normalized

    def fetch_odds_history_15m(self, market_id: str) -> List[OddsSnapshot]:
        """Fetch 15-minute odds snapshots for a market id."""
        history_payload = self._get_json(
            self.clob_base_url,
            "/prices-history",
            {"market": market_id, "interval": "15m"},
        )

        rows = history_payload.get("history", history_payload if isinstance(history_payload, list) else [])
        snapshots: List[OddsSnapshot] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ts = row.get("t") or row.get("timestamp") or row.get("time")
            yes = row.get("p") or row.get("price") or row.get("yes")
            no = row.get("no")

            iso_ts = self._to_iso_utc(ts)
            yes_odds = self._safe_float(yes)
            no_odds = self._safe_float(no)
            if yes_odds is not None and no_odds is None:
                no_odds = round(1.0 - yes_odds, 6)

            snapshots.append(OddsSnapshot(timestamp=iso_ts, yes_odds=yes_odds, no_odds=no_odds))

        return snapshots

    @staticmethod
    def _to_iso_utc(ts: Any) -> str:
        if ts is None:
            return ""
        if isinstance(ts, str) and "T" in ts:
            return ts
        try:
            ts_int = int(float(ts))
            if ts_int > 10_000_000_000:
                ts_int //= 1000
            return datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat()
        except Exception:
            return str(ts)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None


class SportsBettingDatasetBuilder:
    def __init__(self, client: PolymarketClient) -> None:
        self.client = client

    def build(self, desired_count: int) -> List[SportsBettingEvent]:
        raw_events = self.client.fetch_closed_sports_events(desired_count=desired_count)
        output: List[SportsBettingEvent] = []

        for event in raw_events:
            event_id = str(event.get("id", ""))
            title = str(event.get("title") or event.get("question") or "")
            slug = event.get("slug")
            event_url = f"https://polymarket.com/event/{slug}" if slug else ""
            final_result = self._extract_final_result(event)

            market_id = self._extract_market_id(event)
            odds_history = self.client.fetch_odds_history_15m(market_id) if market_id else []

            output.append(
                SportsBettingEvent(
                    event_id=event_id,
                    title=title,
                    event_url=event_url,
                    final_result=final_result,
                    odds_15m=odds_history,
                )
            )
            time.sleep(self.client.pause_seconds)

        return output

    @staticmethod
    def _extract_market_id(event: Dict[str, Any]) -> str:
        for key in ("market", "marketId", "conditionId"):
            value = event.get(key)
            if value:
                return str(value)

        markets = event.get("markets") or []
        if markets and isinstance(markets, list) and isinstance(markets[0], dict):
            market = markets[0]
            for key in ("id", "marketId", "conditionId"):
                if market.get(key):
                    return str(market[key])
        return ""

    @staticmethod
    def _extract_final_result(event: Dict[str, Any]) -> str:
        for key in ("outcome", "result", "resolvedOutcome", "winner"):
            value = event.get(key)
            if value is not None:
                return str(value)
        return "unknown"


class ParquetWriter:
    """Writes records to parquet when pyarrow is available."""

    @staticmethod
    def write(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError(
                "pyarrow is required to write parquet. Install with `pip install pyarrow` and rerun."
            ) from exc

        records_list = list(records)
        table = pa.Table.from_pylist(records_list)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of closed sports events to fetch")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/polymarket_sports_events.parquet"),
        help="Output parquet file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = PolymarketClient()
    builder = SportsBettingDatasetBuilder(client)
    events = builder.build(desired_count=args.count)

    ParquetWriter.write((event.to_record() for event in events), args.output)
    print(f"Wrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
