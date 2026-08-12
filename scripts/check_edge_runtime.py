from __future__ import annotations

import importlib
import json
import sys
from typing import Any


REQUIRED_MODULES = [
    "torch",
    "ultralytics",
    "cv2",
    "yaml",
    "requests",
    "lap",
]


def main() -> None:
    report = {"ok": True, "modules": {}}
    for module_name in REQUIRED_MODULES:
        report["modules"][module_name] = check_module(module_name)
        if not report["modules"][module_name]["ok"]:
            report["ok"] = False

    torch_info = report["modules"].get("torch", {})
    if torch_info.get("ok"):
        import torch

        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["cuda_device_count"] = int(torch.cuda.device_count())

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        sys.exit(1)


def check_module(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    return {"ok": True, "version": str(getattr(module, "__version__", ""))}


if __name__ == "__main__":
    main()
