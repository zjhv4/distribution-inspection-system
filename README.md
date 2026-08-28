# 配电巡检系统

交付版本：2026.08.28

系统包含人员进入管控和移动式 MCB 状态监测。交付配置使用相对路径，不绑定安装目录、用户名或某张显卡。

## 目录

| 目录 | 内容 |
| --- | --- |
| `configs/site.yaml` | 现场配置 |
| `models/delivery/` | Git LFS 交付模型 |
| `examples/` | 原始测试视频 |
| `artifacts/` | 标注结果视频和验证记录 |
| `scripts/` | 启动、自检和告警服务 |
| `tests/` | 回归测试 |

## 测试视频

- [原始测试视频](examples/intrusion_detection.mp4)
- [人员检测标注结果](artifacts/intrusion_detection_result.mp4)

两个视频均为 1280×720、10 FPS、3 秒。结果视频使用交付版可见光人员模型在 CPU 上生成。

## 安装

```bash
git lfs install
git lfs pull
uv sync --frozen --extra test
uv run python scripts/verify_delivery.py
```

`verify_delivery.py` 会检查配置、模型、测试视频、结果视频和 SHA-256。

## 运行

```bash
TASK=intrusion SOURCE=examples/intrusion_detection.mp4 scripts/run_inspection.sh
```

保存新的标注视频：

```bash
TASK=intrusion SOURCE=video.mp4 scripts/run_inspection.sh \
  --output artifacts/result.mp4
```

`runtime.device` 留空时自动选择可用设备。如需固定使用 CPU 或某张 GPU，在 `configs/site.yaml` 中填写 `cpu`、`0`、`1` 等值。

## 告警服务

默认仅监听本机：

```bash
uv run python scripts/alarm_server.py --port 18088
```

对外监听时必须设置令牌：

```bash
export ALARM_TOKEN='replace-with-a-random-secret'
uv run python scripts/alarm_server.py --host 0.0.0.0 --port 18088
```

## 文档

- [部署手册](docs/部署手册.md)
- [配置说明](docs/配置说明.md)
- [系统说明](docs/系统说明.md)
- [测试记录](docs/测试记录.md)
