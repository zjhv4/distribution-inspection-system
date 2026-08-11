from __future__ import annotations

import argparse

from .config import load_site_config
from .pipeline import run_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="配电室边缘巡检视觉检测")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run edge detection on camera or video")
    run_parser.add_argument("--task", choices=["intrusion", "breaker", "all"], required=True)
    run_parser.add_argument("--source", default="0", help="Camera index or video path")
    run_parser.add_argument("--config", default="configs/site.yaml")
    run_parser.add_argument("--display", action="store_true")
    run_parser.add_argument("--output", default=None, help="Optional annotated MP4 output path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = load_site_config(args.config)
        source: str | int = int(args.source) if str(args.source).isdigit() else args.source
        run_video(source=source, config=config, task=args.task, display=args.display, output=args.output)


if __name__ == "__main__":
    main()
