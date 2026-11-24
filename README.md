# SO-101 Self-Learning Workbench

面向 SO-101、Orange Pi 5 Max 和 GPU 电脑的实验性机器人学习工作台。
项目把中文文字指令、双相机观察、Qwen 视觉监督、受限动作执行、失败恢复和在线误差补偿放进一个可检查的流程。

当前发布的是实验代码。单次方块换位已在真实机械臂上通过终态复拍，过程中使用了断点续跑。稳定叠放和完全由零示范试错获得任务策略仍在开发中。

## 当前功能

| 功能 | 实现与验证 |
| --- | --- |
| 中文文字输入 | 本机网页输入互换或叠放指令 |
| 双相机 | 固定相机与腕部相机轮询预览 |
| 视觉监督 | 本地 llama.cpp 运行用户提供的 Qwen VL 权重，核对物体身份与叠放结果 |
| 换位 | 识别两块、寻找暂放点、三段搬运、落点核验、保存进度与续跑 |
| 叠放 | 夹取核验、底座复核、落点修正、分段松爪和退出流程 |
| 在线学习 | RLS 学习关节跟踪误差的前馈补偿，另有可离线导入的视觉编码器和 GRU 策略定义 |
| 真机控制 | 显式启用后运行，保持关节、行程、高度和扭矩检查 |

换位和叠放流程目前主要使用几何规划、实测参考姿态与反馈修正。Qwen 的角色是监督。RLS 学习补偿参数。视觉策略模块定义网络结构，重新训练流程和权重发布列在开放任务中。当前仓库没有 DreamerV3 训练实现。

## 无硬件体验

安装 [uv](https://docs.astral.sh/uv/) 后，在 Python 3.12 环境运行。

```sh
uv sync --python 3.12 --frozen
uv run python scripts/offline_demo.py
uv run python scripts/check_release.py
uv run python -m unittest discover -s workbench -p 'test_*.py' -v
```

第一个示例用合成的恒定关节扰动演示在线 RLS 更新。输出中的误差来自这个合成系统。真机结果见 [测试记录](docs/results.md)。

## 网页

```sh
uv run python workbench/swap_console.py
```

打开 `http://127.0.0.1:8765/`。默认状态允许查看界面，真实相机需要配置 Orange Pi，机械臂动作需要完成自己的硬件配置并显式启用。

可以输入以下指令。

```text
把橙色和带圆纹的深灰色方块互换
继续上次换位
把橙色方块叠放在灰色方块上面
```

## 硬件配置

桌面电脑负责网页、视觉监督和规划。Orange Pi 负责相机与舵机通信。数据通过 SSH/SFTP 传输，网页绑定本机回环地址。

按照 [安装与标定](docs/setup.md) 创建自己的 profile，并将 `settings.example.json` 复制为 `workbench/settings.local.json`。相机设备、串口、主机和模型位置均可配置。

本项目保留了原实验的姿态数值和数值标定样本，放在 `examples/reference-profile/` 中，供离线阅读与复现算法使用。真实机械臂需要自己的标定与工作区检查。默认检查会拒绝把参考 profile 用于动作。

## 代码

- `workbench/` 包含网页、感知、规划与板端控制脚本。
- `learning/rls.py` 包含从真机实验提取的在线 RLS 补偿器。
- `learning/visual_policy.py` 包含可选的 CNN 视觉过程编码器、GRU 策略与关节保护器，需要 `uv sync --extra policy`。
- `scripts/` 包含离线示例、profile 准备工具和发布检查。
- `evidence/` 保留数值结果和失败记录。
- `docs/tasks/` 包含可发布为 GitHub Issues 的开发任务。

## 开发任务

[完整任务列表](docs/roadmap.md) 优先处理从真实经验更新任务策略、稳定叠放、控制权互斥与动作审计、视觉策略重训练和独立评估。

## 开源依赖

代码按 Apache-2.0 发布。SO-101 URDF 与相关网格来自 TheRobotStudio/SO-ARM100，来源和版本见 [来源说明](docs/provenance.md)。LeRobot、llama.cpp 和外部模型权重遵循各自许可证。

## 相关项目

GitHub 已有 SO-101 实机模仿学习、强化学习和 Qwen 视觉控制项目。比较与链接见 [相关工作](docs/related-projects.md)。
