# 安装与标定

## 运行环境

推荐使用桌面端 Python 3.12、Orange Pi 上 Python 3.10 或更新版本。机械臂控制依赖 LeRobot 的 SO101Follower 和 Feetech 支持。上游 API 会演进，应记录自己使用的 LeRobot 版本，并先完成串口读取与标定检查。

桌面端运行 `uv sync --python 3.12 --frozen`。Orange Pi 可以用同一依赖清单安装，再安装匹配的 `lerobot[feetech]`。

本机配置保存在 `workbench/settings.local.json`，可从仓库根目录的 `settings.example.json` 复制。该文件已被 Git 忽略。环境变量使用 `SO101_` 前缀并覆盖文件配置。

## 标定数据

创建 `profiles/my-arm/`，参考 `examples/reference-profile/` 的数据结构。需要重新得到以下数据。

- `alignment_arm_calibration.json` 来自本机机械臂的 LeRobot 标定结果。
- `paper_plane.json` 保存固定相机像素到桌面的映射，可用 `calibrate_table.py` 辅助处理自己的纸面图片。
- `base_alignment_p4_verified.json` 保存静止接触参考的编码器值。
- `base_alignment_p4_checked.json` 与 `base_alignment_checked.json` 保存编码器零点和纸面到基座的注册结果。
- 三个参考子目录分别保存自己的抓取姿态和通过验收的放置结果。
- `swap_stage1_verified.json` 保存第一段放置的独立验收。

现有硬编码工作区与姿态来自一个实验工位。迁移时需要检查 `swap_skill_plan.py` 的停靠点、`swap_blocks.py` 的关节窗口和观察姿态、`learn_orange_grasp.py` 的中转姿态。它们属于原型的待泛化部分。

完成实机检查后，在自己的 `hardware.json` 中设置 `reviewed_for_this_robot` 为 `true`。然后安装 profile。

```sh
uv run python scripts/prepare_profile.py profiles/my-arm
```

桌面和 Orange Pi 各保留自己的配置文件。在板上配置 `profile_dir` 指向已部署的 profile，并执行同一个准备命令，安装到板上的 `workbench/checks/` 和 `workbench/calibration/`。

## 板端目录与 SSH

将仓库完整复制到 Orange Pi。默认远程目录为 `so101-workbench`，默认 Python 为 `.venv/bin/python`。SSH 从用户家目录执行命令，因此可将桌面配置中的 `board_dir` 设为 `so101-workbench/workbench`，`remote_python` 设为 `../.venv/bin/python`。

先用系统 SSH 连接板子，核对其主机指纹，使其进入自己的 `~/.ssh/known_hosts`。桌面端会使用这个文件核验服务器。

桌面配置中填写 `host`、`user` 和上述两个远程路径。密码只从环境变量 `SO101_SSH_PASSWORD` 读取。可以通过交互提示输入。

```sh
uv run python scripts/console.py
```

板端设置自己的 `serial_port`、`fixed_camera`、`wrist_camera`。这些设备会用于相机共享进程和舵机连接。

## Qwen 视觉监督

准备与系统匹配的 llama.cpp `llama-mtmd-cli`，填写配置中的 `vlm_executable` 和 `vlm_device`。将你有使用权且与运行时兼容的 VL GGUF 和 projector 放在 `vlm_dir`，文件名分别为 `model.gguf` 和 `projector.gguf`。

权重需要单独下载。项目发布的代码不携带 Qwen 权重。使用前检查模型和转换权重的许可证。原型使用过 Qwen3-VL 系列，具体量化与 llama.cpp 版本需要由部署者记录。

## 真实动作

先让人离开工作区并准备可直接切断舵机电源的方式。由于当前原型仍有几何误差和释放失败，运行者需现场观察。

完成自己工位的校准后，在桌面进程的环境中设置 `SO101_ENABLE_MOTION=1`。随后从网页提交明确的指令。板端会再次核对 profile。

```powershell
$env:SO101_ENABLE_MOTION = '1'
uv run python scripts/console.py
```

Linux 或 macOS 可使用以下命令。

```sh
SO101_ENABLE_MOTION=1 uv run python scripts/console.py
```

“本步结束后停止”在已执行的搬运结束并退回后生效。它存在延迟，物理断电仍需能在现场使用。网页刷新、浏览相机和运行离线学习示例只处理界面或数据。
