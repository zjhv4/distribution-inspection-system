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

- [原始入侵测试视频](examples/intrusion_detection.mp4)
- [入侵检测标注结果](artifacts/gddw_intrusion_chain_annotated_20260803.mp4)
- [最新全模型动态验收视频](artifacts/rtx_all_models_annotated.mp4)
- [最新热成像入侵验收视频](artifacts/rtx_intrusion_annotated.mp4)

原始视频及对应标注结果为 1280×720、10 FPS、3 秒。全模型验收视频为 406×720、3 FPS、4 秒；热成像入侵验收视频为 640×512、3 FPS、约 3.3 秒。验收视频直接取自本地正式验收产物，未使用外部演示素材。

## 安装

```bash
git lfs install
git lfs pull
uv sync --frozen --extra test
uv run python scripts/verify_delivery.py
```

`verify_delivery.py` 会检查配置、模型、测试及验收视频和 SHA-256。

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
