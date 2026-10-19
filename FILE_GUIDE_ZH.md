# 文件说明（中文版）

适用于 `deep-learning-training-expert` v028。每个文件的用途如下；阅读时先看入口，再按当前问题查对应内容。

## 入口与判断指南

| 文件 | 通俗说明 |
|---|---|
| [SKILL.md](SKILL.md) | Agent 的总工作指南：先理解训练事实，再决定如何初始化、调参、继续、停止和选模型，并遵守当前任务的权限与输出格式。 |
| [references/factual-reconstruction.md](references/factual-reconstruction.md) | 教 Agent 读懂真实日志：这个指标衡量谁、什么时候测的、对应哪个模型，以及先执行了什么、后来发生了什么。 |
| [references/initialization.md](references/initialization.md) | 指导训练开始时的选择：哪些部分继承了预训练能力、哪些需要重新学习，据此考虑初始参数或起始状态。 |
| [references/diagnosis-and-experiments.md](references/diagnosis-and-experiments.md) | 训练出现问题时，区分数据、实现、优化和目标本身的问题，选择有帮助的调整或小实验，并检查调整是否奏效。 |
| [references/stopping-and-selection.md](references/stopping-and-selection.md) | 分别回答“还值得继续训练吗”和“现有模型该选哪个”。停止训练与选择 checkpoint 是两个独立判断。 |

## 自动核对工具

这些脚本帮助核对事实与文件对应关系；不能替代 Agent 判断原因，也不能保证调参正确。

| 文件 | 通俗说明 |
|---|---|
| [scripts/fact_stage.py](scripts/fact_stage.py) | 把原始记录整理成只读事实材料，并检查事实报告的来源与数字，供正式调参前确认理解是否正确。 |
| [scripts/source_claims.py](scripts/source_claims.py) | 检查引用是否指向输入材料里的真实位置，避免把错误的历史条目当作依据。 |
| [scripts/observation_claims.py](scripts/observation_claims.py) | 检查单项观测的身份：指标、类别、样本范围、统计方式、单位、时间和模型状态是否对应。 |
| [scripts/evidence_claims.py](scripts/evidence_claims.py) | 检查前后或两组比较的说法是否符合已登记的证据，例如有没有把下降读成上升、把别的实验当成当前实验。 |
| [scripts/score_diagnostics.py](scripts/score_diagnostics.py) | 区分分数排序能力、分类阈值和最终分类效果；还可逐点比较哪些错误被纠正、哪些错误新出现。仅使用允许访问的标签。 |
| [scripts/completion_receipt.py](scripts/completion_receipt.py) | 复用已有决策前，检查请求、回答和来源记录是否匹配，避免误用过期或属于别次训练的结果。 |

## Cashew 任务专用配置与知识

这一部分服务于腰果 FADA 项目；其中的类别、数据和执行设置不能直接当作其他任务的默认值。

| 文件 | 通俗说明 |
|---|---|
| [task_profiles/cashew_fada/profile.md](task_profiles/cashew_fada/profile.md) | 腰果任务的主要说明：类别编号、源域模型、损失含义、prior 的用法、自动负样本以及执行边界。 |
| [task_profiles/cashew_fada/runtime_contract.json](task_profiles/cashew_fada/runtime_contract.json) | 用结构化格式记录该项目的执行约定，包括固定类别权重、允许的能力、资源约束和历史设置；实际执行仍以当前任务约定为准。 |
| [task_profiles/cashew_fada/packet_contract.py](task_profiles/cashew_fada/packet_contract.py) | 从任务输入提取控制项、诊断计算等约定，并在单独选模型时收紧输出格式；它不替 Agent 选参数。 |
| [task_profiles/cashew_fada/evidence_registry.py](task_profiles/cashew_fada/evidence_registry.py) | 把腰果项目提供的比较记录转换成统一证据条目，保留指标、样本范围和时间，供核对工具使用。 |
| [task_profiles/cashew_fada/component-loss-paths.md](task_profiles/cashew_fada/component-loss-paths.md) | 说明模型各部分实际参与哪些计算、接收哪些损失的梯度，避免把“加载过参数”误认为“正在参与训练”。 |
| [task_profiles/cashew_fada/initialization-probe-semantics.md](task_profiles/cashew_fada/initialization-probe-semantics.md) | 解释初始化诊断中的步数、采样和状态含义，尤其提醒：在缓存样本上拟合好，不等于目标区腰果识别更准。 |
| [task_profiles/cashew_fada/woody-wetland-confusion.md](task_profiles/cashew_fada/woody-wetland-confusion.md) | 针对腰果与混交植被、湿地区域混淆的专家提示：综合模型、prior、特征与人工错误证据决定怎么调整。 |

## 仓库辅助文件

| 文件 | 通俗说明 |
|---|---|
| [README.md](README.md) | 项目介绍、安装方法、能力边界，以及目前实验结果能支持到什么程度。 |
| [MANIFEST.json](MANIFEST.json) | v028 的 18 个 Skill 文件及其校验值，用于确认核心文件是否与发布版本一致；不包含新增的阅读说明。 |
| [.gitignore](.gitignore) | 告诉 Git 忽略 Python 缓存和环境变量文件，减少无关文件及敏感配置误入仓库的机会。 |
| [FILE_GUIDE_ZH.md](FILE_GUIDE_ZH.md) | 本说明，帮助人快速了解每个文件的用途。 |

建议阅读顺序：`SKILL.md` → 当前问题对应的 `references/` → 使用腰果任务时再看 `profile.md`。其余脚本通常由调用程序使用，无需逐个阅读源码。
