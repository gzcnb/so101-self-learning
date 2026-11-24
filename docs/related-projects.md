# 相关公开项目

本页记录 2026 年 11 月 24 日检索到的相关项目。它们为驱动、视觉监督和机器人学习提供参考，具体的示范要求、硬件条件与效果需以各自公开材料为准。

| 项目 | 与本项目相关的部分 |
| --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot) | SO-101 驱动、数据、策略训练及强化学习基础设施 |
| [atomictan/lerobot-so101](https://github.com/atomictan/lerobot-so101) | 记录 SO-101 模仿学习结果及 SAC/HIL-SERL 工程适配，区分了仿真与实机结果 |
| [RajatDandekar/so101-hilserl-whiteboard](https://github.com/RajatDandekar/so101-hilserl-whiteboard) | 真实 SO-101 的擦白板实验，使用示范和人工介入，记录了收敛限制 |
| [ultrafro/garbagecollector](https://github.com/ultrafro/garbagecollector) | SO-101、边缘计算板与本地 Qwen3-VL，包含视觉目标选择和抓取结果核验 |
| [VIGNESHM-ENGR/CogniBot-Ros2-Vla](https://github.com/VIGNESHM-ENGR/CogniBot-Ros2-Vla) | Qwen3-VL、动作规划、网页和闭环执行组合，公开结果以其项目所述测试环境为准 |
| [DreamerV3](https://github.com/danijar/dreamerv3) | 世界模型强化学习研究代码，可作为后续算法研究参考 |
| [SERL](https://serl-robot.github.io/) | 真实机器人离策略学习、奖励定义与控制基础设施参考 |

本项目关注大模型监督下、零人工任务示范的 SO-101 真机学习。当前贡献包括实验工作台、在线补偿器和失败证据，后续需要接通真实经验、策略更新与技能复用，设计见 [自学习说明](self-learning.md)。这些组件在目标任务上的效果需要独立评估。
