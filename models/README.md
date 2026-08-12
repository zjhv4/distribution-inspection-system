# 模型清单

模型权重通过 Git LFS 管理。克隆仓库时需要启用 Git LFS，并可使用以下 SHA-256 核对文件。

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `intrusion_power_visible_yolo11l.pt` | 真实配电/变电可见光人员检测 | `5141d15a1797e372f41aaa52adb7a1efd9b027a1c8af98cb59acaf876b564903` |
| `intrusion_legacy_mixed_yolo11l.pt` | 旧 CCTV/热成像混合域兼容 | `a72554200809810ad72fc9107baebf2c51c915c9bf6cf912f9edf0c09565b131` |
| `breaker_mobile_types_yolo11s.pt` | 移动画面 MCB/RCD/ISOLATOR 检测 | `c8a32b41d45b42943fc665426b30d8ece4ca3777bdcfca1da87ce493e7278352` |
| `breaker_mcb_state_yolo11s_cls.pt` | MCB 拨杆 OPEN/CLOSED/UNKNOWN 分类 | `4483390aee03ec7815403adeae814651fa58a6722b9d98ef0b584a6c2c0f2cd6` |

`power_visible` 用于普通可见光配电、变电画面，`legacy_mixed` 用于旧 CCTV 或热成像画面。断路器默认先实时识别设备类型，再对检测到的 MCB 判断拨杆状态。
