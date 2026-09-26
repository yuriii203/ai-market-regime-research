from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.ifind_mcp import IfindMcpClient


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    client = IfindMcpClient.from_env()
    result = client.call_tool(
        "news",
        "search_news",
        {
            "query": "近期A股重要政策与财经新闻",
            "time_start": "2026-09-20",
            "time_end": "2026-09-26",
            "size": 1,
        },
    )
    records = result.data.get("data", []) if isinstance(result.data, dict) else []
    print(
        f"OK=True RECORDS={len(records)} "
        f"LOCAL_TRACE={result.trace.trace_id.startswith('ifind-')} "
        f"HASH_LENGTH={len(result.trace.response_hash)} "
        f"PROVIDER_REQUEST_ID={result.trace.provider_request_id}"
    )


if __name__ == "__main__":
    main()
