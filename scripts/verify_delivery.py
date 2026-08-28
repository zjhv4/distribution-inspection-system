from __future__ import annotations

import json
import sys
from pathlib import Path

from edge_inspection.config import load_site_config


def main() -> None:
    root = Path(".")
    if not (root / "pyproject.toml").is_file():
        print("Please run this command from the project root.", file=sys.stderr)
        raise SystemExit(2)

    config = load_site_config(root / "configs/default.yaml")
    model_paths = {
        config.models.intrusion,
        *config.models.intrusion_profiles.values(),
        config.models.breaker,
        config.models.breaker_state_classifier,
    }
    failures = []
    for relative_path in sorted(path for path in model_paths if path):
        path = root / relative_path
        if not path.is_file():
            failures.append(f"missing model: {relative_path}")
            continue
        with path.open("rb") as file:
            header = file.read(64)
        if header.startswith(b"version https://git-lfs.github.com"):
            failures.append(f"Git LFS model not downloaded: {relative_path}")

    report = {
        "ok": not failures,
        "config": "configs/default.yaml",
        "runtime_device": config.runtime.device or "auto",
        "models_checked": len(model_paths),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
