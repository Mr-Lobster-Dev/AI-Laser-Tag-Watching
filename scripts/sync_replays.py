"""Copy new replay files into web/replays/ and rebuild web/replays/index.json.

Usage:
    python scripts/sync_replays.py                      # default source folder
    python scripts/sync_replays.py --src path/to/replays

Only files that are missing or changed are copied. The index lists every
match_*.json.gz in web/replays/ with its final scores, sorted by match number.
"""

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "web" / "replays"
DEFAULT_SRC = Path("/mnt/project-files/laser-tag/replays")
NAME = re.compile(r"^match_(\d+)\.json\.gz$")


def summarize(path: Path, old: dict | None) -> dict:
    stat = path.stat()
    if old and old.get("_size") == stat.st_size and old.get("_mtime") == int(stat.st_mtime):
        return old
    with gzip.open(path, "rt") as fh:
        data = json.load(fh)
    scores = data.get("final_scores", {})
    return {
        "match": int(data.get("match", NAME.match(path.name).group(1))),
        "file": path.name,
        "green": scores.get("green", 0),
        "pink": scores.get("pink", 0),
        "winner": scores.get("winner", "draw"),
        "trained_matches": data.get("trained_matches", 0),
        "shots": len(data.get("shots", [])),
        "tags": sum(1 for s in data.get("shots", []) if s.get("tag")),
        "bytes": stat.st_size,
        "_size": stat.st_size,
        "_mtime": int(stat.st_mtime),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    if args.src.is_dir():
        for src in sorted(args.src.iterdir()):
            if not NAME.match(src.name):
                continue
            dst = DEST / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                copied += 1
    else:
        print(f"Source folder {args.src} not found, only rebuilding the index.")

    index_path = DEST / "index.json"
    old = {}
    if index_path.exists():
        try:
            old = {e["file"]: e for e in json.loads(index_path.read_text())}
        except (ValueError, KeyError):
            old = {}

    entries = []
    for path in sorted(DEST.glob("match_*.json.gz")):
        try:
            entries.append(summarize(path, old.get(path.name)))
        except (OSError, ValueError) as err:  # a half-written file
            print(f"Skipping {path.name}: {err}")
    entries.sort(key=lambda e: e["match"])
    # Keep size/mtime so unchanged files are not re-read next time.
    index_path.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"Copied {copied} file(s). Index lists {len(entries)} match(es).")


if __name__ == "__main__":
    main()
