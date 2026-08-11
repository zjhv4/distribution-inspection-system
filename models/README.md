# 模型清单

模型权重通过 Git LFS 管理。克隆仓库时需要启用 Git LFS，并可使用以下 SHA-256 核对文件。

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `intrusion_power_visible_yolo11l.pt` | 真实配电/变电可见光人员检测 | `9ed60d83f6b685c8fd2f80c8c77b2d9da26cafec28e0c12128e5aee23ea4485a` |
| `intrusion_legacy_mixed_yolo11l.pt` | 旧 CCTV/热成像混合域兼容 | `a72554200809810ad72fc9107baebf2c51c915c9bf6cf912f9edf0c09565b131` |
| `dinov2_vits14.safetensors` | 按资产建立现场正常特征库 | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |

`power_visible` 用于普通可见光配电、变电画面，`legacy_mixed` 用于旧 CCTV 或热成像画面。按摄像头类型在配置文件中选择。断路器状态模型按现场设备族单独配置，不与仓库内的人员模型共用。
