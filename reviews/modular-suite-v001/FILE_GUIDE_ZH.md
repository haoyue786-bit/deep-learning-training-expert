# 候选文件说明（中文）

本表为人类审阅用的一行说明，不是英文全文翻译。部分旧跨主题参考与小工具随专业模块复制，便于模块独立使用；同次投递可去重。这不意味着每个文件都要加载，也不意味着每个模块都要启动Agent。

|审阅／登记文件|用途|
|---|---|
|README.zh-CN.md|解释实际新增专业知识、使用方式与边界。|
|../../candidates/modular-suite-v001/README.md|英文候选入口，说明九项能力、知识范围、实际投递与版本身份。|
|KNOWLEDGE_COVERAGE_ZH.md|按主题说明具体内容、来源和剩余工作。|
|PROFESSIONAL_KNOWLEDGE_SOURCE_MAP.json|绑定知识文件与原文／固定代码／真实案例的摘要及推论范围。|
|BUILD_MANIFEST.json|记录英文Skill文件摘要，核对打包身份；不是效果证明。|
|STATUS_ZH.md|记录本地候选当前状态和历史入口。|
|FILE_GUIDE_ZH.md|本文件：逐文件说明。|

## deep-learning-training-expert

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/SKILL.md)|总体训练管理：维持目标、轨迹和专业知识使用|
|[references/composition.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/references/composition.md)|说明九项能力怎样组成套件、同一Agent怎样共享状态及实际投递正文。|
|[references/knowledge-use-and-navigation.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/references/knowledge-use-and-navigation.md)|按症状找到专业知识，并说明怎样把条件和机制用于当前动作。|
|[scripts/suite_packet.py](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/scripts/suite_packet.py)|把明确选定的英文正文打成单次输入并去重；不选科学方案、不调用模型。|
|[task_profiles/cashew_fada/component-loss-paths.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/component-loss-paths.md)|记录实际模型中哪些组件参与分类或对抗梯度，避免按名字猜测。|
|[task_profiles/cashew_fada/data-labels-and-evidence-roles.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/data-labels-and-evidence-roles.md)|登记真实样本／修订／ID／prior路径及用途，区分各cohort和后续组排除。|
|[task_profiles/cashew_fada/evidence_registry.py](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/evidence_registry.py)|将Cashew不同证据总体分开登记核对，不从无标签代理计算目标精度。|
|[task_profiles/cashew_fada/historical-intervention-cases.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/historical-intervention-cases.md)|保留四个原始历史案例的真实响应、反例、限制和文件摘要。|
|[task_profiles/cashew_fada/initialization-probe-semantics.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/initialization-probe-semantics.md)|解释已有初始化探针的计步、拟合集与验证集、准备与采用语义。|
|[task_profiles/cashew_fada/packet_contract.py](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/packet_contract.py)|解读当前Cashew输入控制、探针算术和选模schema；不选择科学动作。|
|[task_profiles/cashew_fada/profile.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/profile.md)|Cashew领域入口：类别、源权重、实际组件、监督、prior与执行约定。|
|[task_profiles/cashew_fada/runtime_contract.json](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/runtime_contract.json)|声明项目示例的类别、固定源目标与执行分工，实际packet决定能力。|
|[task_profiles/cashew_fada/tile-evidence-to-training-choices.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/tile-evidence-to-training-choices.md)|把tile证据与候选初始化／调整方向连接，不虚构最优参数函数。|
|[task_profiles/cashew_fada/woody-wetland-confusion.md](../../candidates/modular-suite-v001/skills/deep-learning-training-expert/task_profiles/cashew_fada/woody-wetland-confusion.md)|保留用户批准的混交林／湿地混淆与区域常识判断。|

## training-adjustment

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-adjustment/SKILL.md)|按当前机制选择或撤回实际训练改动|
|[references/adversarial-objectives-and-timescales.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/adversarial-objectives-and-timescales.md)|深入解释GAN／FADA／CDAN、D速率和频次、TTUR、谱归一化及交互。|
|[references/adversarial-practice.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/adversarial-practice.md)|对抗训练的跨主题说明，区分对手能力、条件语义和梯度机制。|
|[references/class-bias-and-pseudo-label-correction.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/class-bias-and-pseudo-label-correction.md)|详解伪类偏差、DebiasPL相反符号、CCA-LSC比例修正及错误放大条件。|
|[references/diagnosis-and-experiments.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/diagnosis-and-experiments.md)|解释比较作用域、可区分假设、干预响应、反证及恢复。|
|[references/optimization-practice.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/optimization-practice.md)|优化问题的跨主题参考，连接更新、泛化、组件及交互设置。|
|[references/optimizer-state-and-control-interactions.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/optimizer-state-and-control-interactions.md)|深入解释动量、Adam历史、epsilon、AdamW衰减、分组和实际位移。|
|[references/pseudo-label-confidence-and-feedback.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/pseudo-label-confidence-and-feedback.md)|说明教师、置信、一致性、熵与弱标签如何形成正确或错误反馈。|
|[references/regularization-sampling-and-loss-design.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/regularization-sampling-and-loss-design.md)|说明正则、采样、类别权重、loss归一化、增强和有效曝光。|
|[references/schedules-batches-and-numerical-stability.md](../../candidates/modular-suite-v001/skills/training-adjustment/references/schedules-batches-and-numerical-stability.md)|解释日程、有效更新、batch／累积、warmup、裁剪及AMP顺序。|
|[scripts/score_diagnostics.py](../../candidates/modular-suite-v001/skills/training-adjustment/scripts/score_diagnostics.py)|计算受支持的排序／阈值或逐点错误比较，不替Agent决定目标。|

## training-checkpoint-selection

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-checkpoint-selection/SKILL.md)|比较已有模型及完整部署策略|
|[references/checkpoint-ranking-calibration-and-deployment.md](../../candidates/modular-suite-v001/skills/training-checkpoint-selection/references/checkpoint-ranking-calibration-and-deployment.md)|详解选模、排序／校准／阈值、label shift、拒识覆盖及部署身份。|
|[references/factual-reconstruction.md](../../candidates/modular-suite-v001/skills/training-checkpoint-selection/references/factual-reconstruction.md)|说明事实、代码推断、历史执行、时序与比较身份的区别。|
|[references/stopping-and-selection.md](../../candidates/modular-suite-v001/skills/training-checkpoint-selection/references/stopping-and-selection.md)|STOP和SELECT的跨主题原则，强调未来训练与已有模型比较的区别。|
|[scripts/fact_stage.py](../../candidates/modular-suite-v001/skills/training-checkpoint-selection/scripts/fact_stage.py)|按已提供结构恢复事实字段及阶段；不根据字段自动判断训练质量。|
|[scripts/source_claims.py](../../candidates/modular-suite-v001/skills/training-checkpoint-selection/scripts/source_claims.py)|将结论用到的数值绑定到原始来源路径和身份。|

## training-diagnosis

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-diagnosis/SKILL.md)|区分数据、实现、优化及语义迁移问题|
|[references/adversarial-practice.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/adversarial-practice.md)|对抗训练的跨主题说明，区分对手能力、条件语义和梯度机制。|
|[references/diagnosis-and-experiments.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/diagnosis-and-experiments.md)|解释比较作用域、可区分假设、干预响应、反证及恢复。|
|[references/domain-shift-and-negative-transfer.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/domain-shift-and-negative-transfer.md)|解释类别混合、条件分布、support和先验估计，识别负迁移路径。|
|[references/multiple-losses-and-gradient-conflict.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/multiple-losses-and-gradient-conflict.md)|解释共享梯度、GradNorm、PCGrad、不确定性加权及各自反例。|
|[references/optimization-practice.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/optimization-practice.md)|优化问题的跨主题参考，连接更新、泛化、组件及交互设置。|
|[references/remote-sensing-shift-and-context.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/remote-sensing-shift-and-context.md)|区分遥感输入、物理范围和语义变化，说明Fourier／农业Swin经验与边界。|
|[references/symptoms-and-differential-diagnosis.md](../../candidates/modular-suite-v001/skills/training-diagnosis/references/symptoms-and-differential-diagnosis.md)|按症状组合鉴别拟合、实现、数据、泛化与遗忘问题。|
|[scripts/score_diagnostics.py](../../candidates/modular-suite-v001/skills/training-diagnosis/scripts/score_diagnostics.py)|计算受支持的排序／阈值或逐点错误比较，不替Agent决定目标。|

## training-experience-update

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-experience-update/SKILL.md)|从真实响应提炼并修正条件经验|
|[references/conditional-experience.md](../../candidates/modular-suite-v001/skills/training-experience-update/references/conditional-experience.md)|经验学习的简短导读，提醒适用性和反思解释的范围。|
|[references/transferable-experience-and-revision.md](../../candidates/modular-suite-v001/skills/training-experience-update/references/transferable-experience-and-revision.md)|详解机制匹配、TNP／Reflexion／ExpeL／A-Mem及正负空效应的经验修正。|

## training-experiment

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-experiment/SKILL.md)|设计并解释有辨识力的具体对照|
|[references/diagnosis-and-experiments.md](../../candidates/modular-suite-v001/skills/training-experiment/references/diagnosis-and-experiments.md)|解释比较作用域、可区分假设、干预响应、反证及恢复。|
|[references/mechanism-directed-training-procedures.md](../../candidates/modular-suite-v001/skills/training-experiment/references/mechanism-directed-training-procedures.md)|保留九类具体干预方法，并解释各自问题、测量、有效与无效响应。|
|[scripts/score_diagnostics.py](../../candidates/modular-suite-v001/skills/training-experiment/scripts/score_diagnostics.py)|计算受支持的排序／阈值或逐点错误比较，不替Agent决定目标。|

## training-fact-reconstruction

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/SKILL.md)|恢复日志中的事实、指标含义和训练状态|
|[references/factual-reconstruction.md](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/references/factual-reconstruction.md)|说明事实、代码推断、历史执行、时序与比较身份的区别。|
|[references/metrics-populations-and-error-slices.md](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/references/metrics-populations-and-error-slices.md)|解释F1、聚合、loss分母、排序、概率与空间样本的实际含义。|
|[references/training-state-and-trajectory-identity.md](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/references/training-state-and-trajectory-identity.md)|区分新起点、权重热启动、完整续训和分支采用，恢复模型及优化状态身份。|
|[scripts/evidence_claims.py](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/scripts/evidence_claims.py)|核对提供的证据声明与源记录，不能证明因果解释。|
|[scripts/fact_stage.py](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/scripts/fact_stage.py)|按已提供结构恢复事实字段及阶段；不根据字段自动判断训练质量。|
|[scripts/observation_claims.py](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/scripts/observation_claims.py)|核对观测名称、类别、范围和方向等声明。|
|[scripts/source_claims.py](../../candidates/modular-suite-v001/skills/training-fact-reconstruction/scripts/source_claims.py)|将结论用到的数值绑定到原始来源路径和身份。|

## training-initialization

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-initialization/SKILL.md)|选择合理起点、组件准备和初始优化策略|
|[references/functional-starts-and-transfer.md](../../candidates/modular-suite-v001/skills/training-initialization/references/functional-starts-and-transfer.md)|详细解释fan尺度、残差零出口、头部偏置、采样和加权归一化及迁移起点。|
|[references/initialization.md](../../candidates/modular-suite-v001/skills/training-initialization/references/initialization.md)|初始化的跨主题说明：组件活动、起点身份及探针的适用范围。|
|[references/normalization-freezing-and-domain-mixtures.md](../../candidates/modular-suite-v001/skills/training-initialization/references/normalization-freezing-and-domain-mixtures.md)|解释冻结参数、buffer、train/eval及跨域混合为什么不同。|
|[references/optimization-practice.md](../../candidates/modular-suite-v001/skills/training-initialization/references/optimization-practice.md)|优化问题的跨主题参考，连接更新、泛化、组件及交互设置。|
|[references/transformer-and-segmentation-transfer.md](../../candidates/modular-suite-v001/skills/training-initialization/references/transformer-and-segmentation-transfer.md)|说明Swin／MiT／分割组件、多尺度、差异学习率及DAFormer／MIC机制。|

## training-stop

|文件|用途|
|---|---|
|[SKILL.md](../../candidates/modular-suite-v001/skills/training-stop/SKILL.md)|判断还有什么有价值的训练机会|
|[references/optimization-practice.md](../../candidates/modular-suite-v001/skills/training-stop/references/optimization-practice.md)|优化问题的跨主题参考，连接更新、泛化、组件及交互设置。|
|[references/stopping-and-selection.md](../../candidates/modular-suite-v001/skills/training-stop/references/stopping-and-selection.md)|STOP和SELECT的跨主题原则，强调未来训练与已有模型比较的区别。|
|[references/training-sufficiency-and-recovery.md](../../candidates/modular-suite-v001/skills/training-stop/references/training-sufficiency-and-recovery.md)|深入解释平台、迟发响应、日程耗尽、EMA时标、恢复和曲线外推。|
