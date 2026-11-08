#!/usr/bin/env python3
"""Build an offline cache from the public MedlinePlus health-topic service."""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.medical_kb import KB_PATH, TOPIC_SPECS


API_ROOT = "https://wsearch.nlm.nih.gov/ws/query"
TOOL_NAME = "elderly-depression-research"


def clean_html(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</p\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def fetch_topic(query: str) -> tuple[str, str, str, str]:
    params = urllib.parse.urlencode(
        {
            "db": "healthTopics",
            "term": query,
            "retmax": "3",
            "tool": TOOL_NAME,
        }
    )
    request = urllib.request.Request(
        f"{API_ROOT}?{params}",
        headers={"User-Agent": "ElderlyDepressionResearch/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    document = root.find(".//document")
    if document is None:
        raise RuntimeError(f"No MedlinePlus topic for query: {query}")
    contents = {}
    for node in document.findall("content"):
        name = node.attrib.get("name", "")
        if name and name not in contents:
            contents[name] = clean_html(node.text or "")
    title = contents.get("title", query)
    summary = contents.get("FullSummary") or contents.get("snippet", "")
    organization = contents.get("organizationName", "National Library of Medicine")
    url = document.attrib.get("url", "")
    if not summary:
        raise RuntimeError(f"No summary returned for query: {query}")
    return title, summary[:1600], organization, url


def main() -> None:
    topics = []
    for index, spec in enumerate(TOPIC_SPECS):
        try:
            title, summary, organization, url = fetch_topic(spec["query"])
        except Exception as exc:
            print(f"[skip] {spec['id']}: {exc}")
            continue
        topics.append(
            {
                "id": spec["id"],
                "title": title,
                "title_zh": spec["title_zh"],
                "summary": summary,
                "organization": organization,
                "url": url,
                "triggers": spec["triggers"],
            }
        )
        print(f"[ok] {spec['id']}: {title}")
        if index < len(TOPIC_SPECS) - 1:
            time.sleep(0.8)
    if not topics:
        raise SystemExit("No MedlinePlus topics were downloaded")
    payload = {
        "source": "MedlinePlus.gov",
        "organization": "U.S. National Library of Medicine",
        "citation": (
            "MedlinePlus [Internet]. Bethesda (MD): National Library of "
            "Medicine (US). Available from: https://medlineplus.gov/"
        ),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "topic_count": len(topics),
        "topics": topics,
    }
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    KB_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(topics)} topics to {KB_PATH}")


if __name__ == "__main__":
    main()
