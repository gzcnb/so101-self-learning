# 代码与数据来源

## 项目代码

`workbench/` 来自这一轮 SO-101 实机实验的软件原型。发布副本提取了可复用模块，移除了旧的一次性抓取尝试入口，并将主机、设备、模型和数据位置改为本机配置。

`learning/rls.py` 的补偿器从原 `bio_proprio_exp.py` 中提取，保留其更新规则。`learning/visual_policy.py` 来自 `real_model_r1/model.py`，保留编码器、策略与保护器定义。

原工位 profile 中的数值参考属于实验样本。摄像头序列号、网络地址、口令、SSH 主机密钥、私人录像、机器学习权重均未包含在这个公开代码包中。

## 机器人几何

来源为 [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)，快照提交为 `141aea7bcac2744d92998e30268c413202a56f96`。

保留了 `Simulation/SO101/so101_new_calib.urdf` 及该文件引用的网格，按原 Apache-2.0 许可证分发。文件放在 `workbench/geometry/` 与其相对引用目录。

## 第三方运行时

LeRobot 作为外部依赖使用。llama.cpp 可执行文件和 Qwen 模型权重由部署者自行取得，并遵循各自许可证。PyTorch 仅用于可选视觉策略模块。

## 证据

`evidence/` 来自该工位的数值测试摘要，删除了本机路径、模型路径和未参与结果核验的潜变量。保留了失败结果，供使用者判断当前能力。
