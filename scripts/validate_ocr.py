from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine, text


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def cer(expected: str, actual: str) -> float:
    expected = expected.strip()
    actual = actual.strip()
    if not expected:
        return 0.0 if not actual else 1.0
    return levenshtein(expected, actual) / len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--album-id", required=True)
    parser.add_argument("--ground-truth", type=Path, default=Path("data/golden_fixtures/ocr_ground_truth.json"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--output", type=Path, default=Path("data/golden_ocr_report.json"))
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")

    expected = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    engine = create_engine(args.database_url.replace("+asyncpg", "").replace("+aiosqlite", ""))
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT p.original_filename, string_agg(o.text_content, ' ' ORDER BY o.confidence DESC NULLS LAST) AS text
                FROM pages p
                LEFT JOIN ocr_results o ON o.page_id = p.id
                WHERE p.album_id = :album_id
                GROUP BY p.original_filename
                """
            ),
            {"album_id": str(UUID(args.album_id))},
        ).mappings().all()
    actual = {row["original_filename"]: row["text"] or "" for row in rows}

    reports = []
    all_cer = []
    for filename, expected_text in expected.items():
        score = cer(expected_text, actual.get(filename, ""))
        all_cer.append(score)
        reports.append({"fixture": filename, "cer": score, "expected": expected_text, "actual": actual.get(filename, "")})

    mean_cer = sum(all_cer) / len(all_cer) if all_cer else 1.0
    payload = {"mean_cer": mean_cer, "fixtures": reports}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if mean_cer > 0.10:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
