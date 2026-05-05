"""Repair a truncated eval_results JSON file in place.

Walks back to the last complete entry boundary (`\\n    },\\n`), closes the
outer dict, writes back. Backs up the original to <path>.truncated.bak.
"""

import json
import sys
from pathlib import Path


def repair(path: Path):
    raw = path.read_text()
    print(f"original size: {len(raw)} bytes")

    try:
        d = json.loads(raw)
        print(f"file already parses cleanly ({len(d)} entries) — nothing to do")
        return
    except json.JSONDecodeError as e:
        print(f"parse error at line {e.lineno} col {e.colno}: {e.msg}")

    marker = "\n    },\n"
    idx = raw.rfind(marker)
    if idx < 0:
        raise RuntimeError("no entry-boundary marker found — file is too damaged to repair")

    repaired = raw[: idx + len("\n    }")] + "\n}\n"
    d = json.loads(repaired)
    print(f"repaired: {len(d)} entries recoverable")

    backup = path.with_suffix(path.suffix + ".truncated.bak")
    backup.write_text(raw)
    print(f"backed up original to {backup}")

    path.write_text(repaired)
    print(f"wrote repaired file to {path} ({len(repaired)} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/eval_results.json>")
        sys.exit(2)
    repair(Path(sys.argv[1]))
