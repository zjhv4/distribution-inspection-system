# 配电巡检系统

用于配电场景中的人员进入管控和断路器事件处理。

## 快速开始

项目使用 Git LFS 存放模型。首次拉取后先确认模型不是 LFS 指针文件：

```bash
git lfs install
git lfs pull
uv sync --frozen --extra test
uv run python scripts/verify_delivery.py
```

运行示例视频：

```bash
TASK=intrusion SOURCE=examples/intrusion_detection.mp4 scripts/run_inspection.sh
```

直接使用 CLI 时，默认读取 `configs/default.yaml`：

```bash
uv run python -m edge_inspection run \
  --task intrusion \
  --source examples/intrusion_detection.mp4
```

`runtime.device` 留空时自动选择可用设备。如需固定使用 CPU 或某张 GPU，在配置中显式填写 `cpu`、`0`、`1` 等值，启动脚本不再强制绑定显卡。

## 人员进入管控

系统对画面中的人员持续跟踪，并结合电子围栏、身份权限和授权时段判断是否需要告警。

- 已授权且处于允许时段：正常通行；
- 无进入权限：产生进入告警；
- 超出授权时段：产生超时进入告警；
- 暂时无法确认身份：进入待复核流程。

普通可见光画面和旧 CCTV、热成像画面分别使用对应模型，避免不同成像域混用造成误判。

## 断路器状态监测

断路器模块面向移动巡检画面，使用同一份设备模型实时识别远近景中的 `MCB`、`RCD` 和 `ISOLATOR`，再由独立状态模型读取 MCB 拨杆位置。设备不在画面中时不显示标签；拨杆被遮挡或看不清时输出 `UNKNOWN`。`TRIP` 与 `MICRO_TRIP` 根据同一设备的连续状态生成，不作为单帧图像类别。

- 设备连续保持稳定闭合后，状态监测进入布防；
- 高置信短时偏离后恢复：生成 `MICRO_TRIP`；
- 高置信偏离达到跳闸确认时间：生成 `TRIP`；
- 状态恢复后：生成同一事件的 `RECOVERED`。

低置信结果、时间戳间断和未完成布防的状态变化不会累计为事件。系统支持结合保护动作、跳闸线圈或辅助触点信号进行多源确认。

## 测试结果

- 可见光人员模型独立测试 Precision 为 0.966，Recall 为 0.907，mAP50 为 0.921；
- 断路器设备模型测试集 Precision 为 0.981，Recall 为 0.981，mAP50 为 0.990；
- MCB 状态模型测试集 Top-1 准确率为 0.930；
- 人员进入、跳闸和微跳事件状态机均已完成告警链路验证；
- 告警投递测试无重复入库，失败任务可在程序重启后继续处理；
- 核心回归测试全部通过。

## 告警服务

默认只监听本机：

```bash
uv run python scripts/alarm_server.py --port 18088
```

如需监听非本机地址，必须设置访问令牌。推理程序会自动从同一环境变量中取出令牌发送给服务端：

```bash
export ALARM_TOKEN='replace-with-a-random-secret'
uv run python scripts/alarm_server.py --host 0.0.0.0 --port 18088
```

`GET /health` 不需要令牌，告警查询、确认和写入接口需要 `Authorization: Bearer <token>`。

## 验证

```bash
uv run pytest
uv run python -m compileall -q edge_inspection scripts tests
uv build
```

详细结果见[测试记录](docs/测试记录.md)，配置方法见[配置说明](docs/配置说明.md)。
