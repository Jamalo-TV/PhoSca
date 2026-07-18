"""Create deterministic reviewed-dataset splits without album/source leakage."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")


def read_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    seen_pages: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        page_id = str(record.get("page_id") or "")
        album_id = str(record.get("album_id") or "")
        source_group = str(record.get("source_group") or "")
        if not page_id or not album_id or not source_group:
            raise ValueError(f"line {line_number} must contain page_id, album_id, and source_group")
        if page_id in seen_pages:
            raise ValueError(f"duplicate page_id in manifest: {page_id}")
        seen_pages.add(page_id)
        records.append(record)
    if not records:
        raise ValueError("reviewed manifest is empty")
    return records


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _leakage_components(records: list[dict]) -> list[list[dict]]:
    union = _UnionFind()
    for record in records:
        union.union(f"album:{record['album_id']}", f"source:{record['source_group']}")

    components: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        components[union.find(f"album:{record['album_id']}")].append(record)

    hashes: dict[str, str] = {}
    for root, component in components.items():
        for record in component:
            image_hash = str((record.get("image") or {}).get("sha256") or "")
            if not image_hash:
                raise ValueError(f"page {record['page_id']} is missing image.sha256")
            previous = hashes.setdefault(image_hash, root)
            if previous != root:
                raise ValueError(f"image hash {image_hash} occurs in different album/source groups")
    return list(components.values())


def split_records(
    records: list[dict],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[dict[str, list[dict]], dict]:
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    if any(value <= 0.0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("train/validation/test ratios must be positive and sum to 1")
    components = _leakage_components(records)
    if len(components) < len(SPLITS):
        raise ValueError(
            f"at least {len(SPLITS)} independent album/source groups are required; found {len(components)}"
        )

    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    result: dict[str, list[dict]] = {name: [] for name in SPLITS}
    assigned_components: dict[str, list[list[dict]]] = {name: [] for name in SPLITS}
    # Seed every requested split before balancing remaining groups.
    for name, component in zip(SPLITS, components[: len(SPLITS)]):
        result[name].extend(component)
        assigned_components[name].append(component)
    targets = {name: ratios[name] * len(records) for name in SPLITS}
    for component in components[len(SPLITS) :]:
        destination = max(SPLITS, key=lambda name: targets[name] - len(result[name]))
        result[destination].extend(component)
        assigned_components[destination].append(component)

    album_sets = {name: {str(record["album_id"]) for record in values} for name, values in result.items()}
    source_sets = {name: {str(record["source_group"]) for record in values} for name, values in result.items()}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            if album_sets[left] & album_sets[right]:
                raise AssertionError(f"album leakage between {left} and {right}")
            if source_sets[left] & source_sets[right]:
                raise AssertionError(f"source-group leakage between {left} and {right}")

    for values in result.values():
        values.sort(key=lambda record: str(record["page_id"]))
    report = {
        "schema_version": 1,
        "seed": seed,
        "ratios": ratios,
        "counts": {name: len(result[name]) for name in SPLITS},
        "albums": {name: sorted(album_sets[name]) for name in SPLITS},
        "source_groups": {name: sorted(source_sets[name]) for name in SPLITS},
        "page_ids": {name: [str(record["page_id"]) for record in result[name]] for name in SPLITS},
    }
    return result, report


def write_splits(output_dir: Path, splits: dict[str, list[dict]], report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        temporary = output_dir / f".{name}.jsonl.tmp"
        temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
        temporary.replace(output_dir / f"{name}.jsonl")
    (output_dir / "split-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split reviewed manifests while keeping albums/source groups isolated."
    )
    parser.add_argument("--input", type=Path, default=Path("data/reviewed_dataset/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reviewed_dataset/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    splits, report = split_records(
        read_manifest(args.input),
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    write_splits(args.output, splits, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
