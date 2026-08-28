from edge_inspection.cli import build_parser


def test_cli_default_config_exists() -> None:
    args = build_parser().parse_args(["run", "--task", "intrusion"])
    assert args.config == "configs/default.yaml"
