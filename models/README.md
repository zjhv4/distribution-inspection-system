# 模型清单

模型权重通过 Git LFS 管理。克隆仓库时需要启用 Git LFS，并可使用以下 SHA-256 核对文件。

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `intrusion_power_visible_yolo11l.pt` | 真实配电/变电可见光人员检测 | `5141d15a1797e372f41aaa52adb7a1efd9b027a1c8af98cb59acaf876b564903` |
| `intrusion_legacy_mixed_yolo11l.pt` | 旧 CCTV/热成像混合域兼容 | `a72554200809810ad72fc9107baebf2c51c915c9bf6cf912f9edf0c09565b131` |
| `breaker_mcb_closed_open_yolo11s_cls.pt` | 低压微型断路器 CLOSED/OPEN 状态分类 | `f150a7697f7918226e166a7a9a6c3eaf05ddd26253b0c54da924e82e67594a93` |
| `dinov2_vits14.safetensors` | 按资产建立现场正常特征库 | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |

`power_visible` 用于普通可见光配电、变电画面，`legacy_mixed` 用于旧 CCTV 或热成像画面。`breaker_mcb_closed_open_yolo11s_cls.pt` 用于低压微型断路器固定 ROI 的开合状态判定。
