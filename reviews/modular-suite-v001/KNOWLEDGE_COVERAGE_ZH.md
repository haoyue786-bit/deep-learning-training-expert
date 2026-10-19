# 专家行为与专业知识共同建构：覆盖清单

最新更新：2026-10-19。首要标准是深入理解人类专业知识，并让后续Agent在具体问题中找得到、用得上、用得对。下列均为本地英文候选资料，正式安装版未替换。当前不启动新训练、科学模型评测或公开benchmark。

当前是知识建设工作区，不是效果验收报告。用户明确要求行为像人类专家，同时拥有支撑该判断的专业知识；两者缺一不可。入口和目录完成不代表知识完备。

## 本轮已经写入Agent可读资料的实质内容

|主题|具体新增知识|对应资料|
|---|---|---|
|功能性初始化|fan尺度与适用假设、整网置零与残差出口置零的区别、新旧组件信号关系、采样／类别权重影响分类偏置、头部准备的收益与失败情境|training-initialization/references/functional-starts-and-transfer.md|
|优化器机制|动量／Adam历史、beta与目标变化、epsilon、全loss缩放与辅助loss缩放的差别、AdamW累计衰减、零梯度／None／冻结、解冻与状态加载|training-adjustment/references/optimizer-state-and-control-interactions.md|
|日程及数值|真实更新与epoch、累积尾批、短跑与长跑前缀、warmup成本、不同日程、batch统计与吞吐、裁剪和AMP的实际次序、三种不稳定机制实例|training-adjustment/references/schedules-batches-and-numerical-stability.md|
|归一化及域混合|参数／buffer／模式四种状态、BN与LN区别、额外forward为何改变模型、不同冻结与校准政策的取舍、只恢复buffer的诊断例|training-initialization/references/normalization-freezing-and-domain-mixtures.md|

本轮第二组已经补入：

|主题|具体新增知识|对应资料|
|---|---|---|
|症状鉴别与诊断|实际输入和loss检查、小样本拟合的能力边界、参数不动与拟合差区别、过拟合和域偏移、忘记与负迁移、具体Cashew过分的竞争解释|training-diagnosis/references/symptoms-and-differential-diagnosis.md|
|正则化／采样／损失|不同干预对象、重复加权、weighted CE不同归一化的算例、真实crop曝光、RCS共现、dropout概率约定和增强语义|training-adjustment/references/regularization-sampling-and-loss-design.md|
|伪标签反馈|teacher／置信／一致性的不同证据、DAFormer整batch广播权重、SENTRY选择性熵、正反馈错误、独立弱负样本路线及总贡献|training-adjustment/references/pseudo-label-confidence-and-feedback.md|
|指标与错误切片|混淆计数、无正样本时F1和recall的区别、聚合、逐点错误变化、排序／概率／解码分离、空间依赖及loss分母|training-fact-reconstruction/references/metrics-populations-and-error-slices.md|

第三组已补入：

|主题|具体新增知识|对应资料|
|---|---|---|
|对抗目标与时间尺度|logistic G目标导数、D／model梯度路径、D拟合与泛化、LR／cadence／lambda作用区别、TTUR反例、FADA与CDAN条件机制、稳定化方法差异|training-adjustment/references/adversarial-objectives-and-timescales.md|
|域偏移与负迁移|边缘与类别条件分布、比例与support区别、混淆矩阵先验估计的两种定义、不可识别情况、缺席类别、先验与弱负样本的语义|training-diagnosis/references/domain-shift-and-negative-transfer.md|
|多损失与梯度冲突|共享参数对齐、loss／梯度／实际更新区别、GradNorm目标及失败、PCGrad投影及停滞算例、对抗目标与协作任务的不同|training-diagnosis/references/multiple-losses-and-gradient-conflict.md|

上述知识均已接入相应SKILL.md的具体读取入口；来源和局部推导范围见PROFESSIONAL_KNOWLEDGE_SOURCE_MAP.json。公式和演示例不是新的训练结果。现有原始调用快照及正式安装版未改。

第四组已经写入并接通入口：

|主题|真正新增的知识与方法|对应资料|
|---|---|---|
|Transformer／分割迁移|Swin与MiT区别，窗口／分辨率／特征金字塔，继承组件与新head的不同优化，DAFormer三种机制及MIC上下文收益和失败条件|training-initialization/references/transformer-and-segmentation-transfer.md|
|训练充分性／恢复|多种平台机制、实际更新与有效类别曝光、EMA响应时标、日程和训练长度耦合、继续／重规划／完整恢复差异、曲线外推的模型假设|training-stop/references/training-sufficiency-and-recovery.md|
|选模／阈值／校准|固定规则与拟合策略的比较，排序／校准／argmax区别，多分类温度导致跨样本排序反转的算例，MLLS条件、拒识覆盖及无标签选择|training-checkpoint-selection/references/checkpoint-ranking-calibration-and-deployment.md|
|具体干预方法|保留并展开九类旧干预：lambda、模型LR、D的LR／频次、温度、置信cap、两种衰减、固定特征D拟合；逐项说明适用现象、比较、测量、响应解释和无效情况|training-experiment/references/mechanism-directed-training-procedures.md|
|经验迁移与修正|TNP的响应相关迁移、Reflexion反馈局限、ExpeL经验与案例、A-Mem原事实和解释分离；按机制匹配，正／负／空效应分别提炼和撤回|training-experience-update/references/transferable-experience-and-revision.md|
|Cashew特征与动作联系|继承误差、新D准备、有效正样本曝光、混交林／湿地、区域prior与原始AEF距离怎样改变初始化／调整方向，独立低权重负样本的累计贡献|deep-learning-training-expert/task_profiles/cashew_fada/tile-evidence-to-training-choices.md|
|真实历史经验|扩查2,096份comparison，1,479份可汇总单tile对照、239个不同tile；条件分层、同tile重复归并、保留相反响应，收窄单例LR经验；原192目标F1动作子集另报，不把机械效果当质量收益|deep-learning-training-expert/task_profiles/cashew_fada/historical-intervention-cases.md；MULTITILE_EXPERIENCE_REVIEW_ZH.md|
|知识利用入口|按实际症状找到机制资料；区分数学、作者代码、论文经验、本地推论和项目合同；把资料条件落实到动作与后续响应，而非只引用标题|deep-learning-training-expert/references/knowledge-use-and-navigation.md|

另外扩充既有两项资料：多损失中加入不确定性加权的似然、分类近似、动态权重与日程交互；对抗知识中加入谱归一化的参数化、近似误差、卷积算子边界及D频次交互。没有将新算法自动安装到训练器。

第五组完成知识与入口的交叉整合：

|主题|具体内容|对应资料|
|---|---|---|
|伪类偏差修正|预测频率不等于真实比例，DebiasPL伪标签／学生margin相反符号、EMA状态，CCA-LSC有界比例修正和作者代码差异|training-adjustment/references/class-bias-and-pseudo-label-correction.md|
|训练状态身份|权重热启动、完整续训、分支／ donor采用，模型／D／优化器／日程／教师／数据与随机状态，最佳状态引用和轨迹混淆|training-fact-reconstruction/references/training-state-and-trajectory-identity.md|
|遥感输入与情境|物理分辨率、crop范围、类别语义、类感知Fourier、两个D的真实分工、农业Swin precision／recall取舍、熵排序与评价聚合|training-diagnosis/references/remote-sensing-shift-and-context.md|
|项目数据与证据角色|cashew.kmz与不同人工bank，当前样本修订／跨年ID、prior版本、CIV后续组排除；逐点目标类别不写入初始化知识|deep-learning-training-expert/task_profiles/cashew_fada/data-labels-and-evidence-roles.md|

八项专业入口的读取顺序已调整：相关深入机制资料优先，旧跨主题短参考留作按需上下文。补强有限batch加权CE的严格先验表达及单样本反例，避免仅以“分母不依赖预测”误用q×w公式。数学、原文、代码、项目合同及解释的范围保持区分。

中文[专业审阅说明](README.zh-CN.md)解释实际内容，另有[逐文件用途](FILE_GUIDE_ZH.md)。这些均不是全量逐字翻译。

## 整体内容进度（本表覆盖早先的待办状态）

|知识板块|本轮已建内容|核查后的适用范围与维护重点|
|---|---|---|
|任务／数据／指标|专业参考、算术及错误切片已建立，与选模接通|Cashew producer的loss定义独立保留，使用时保持实际总体语义|
|初始化／组件／冻结|功能初始化、模式buffer、组件准备和模型族已展开|新旧头、残差出口、BN／LN与有限batch公式分别适用，避免混用配方|
|优化器／参数关系|Adam／SGD／AdamW状态、衰减、分组及更新差异已展开|实际版本、梯度子集及当前状态决定解释范围|
|日程／batch／裁剪／精度|机制及实操已建立，并接通STOP时机|平台前后日程与恢复实例已接入；改变horizon不是简单延长原前缀|
|欠拟合／过拟合／不稳定鉴别|鉴别表、任务错误与机械响应已展开|同一症状保留不同可行动解释，参考正文实例选择诊断方向|
|正则／采样／不平衡／伪标签|损失曝光、置信反馈及类别偏差方法已展开|DebiasPL／CCA-LSC／SENTRY等作用环节、假设和代码差异分开；没有自动新增方法|
|Swin／Transformer／分割|已有独立深入参考及DAFormer／MIC机制|遥感资料补齐物理尺度及Fourier边界，AEF不当光学波段|
|GAN／域对抗|目标、D拟合／泛化、TTUR、SN及交互已展开|不把图像GAN结果误用成无标签目标分割精度|
|多损失／梯度冲突|GradNorm、PCGrad、不确定性加权和反例已建立|共享坐标、似然近似及任务优先级决定能否借用方法|
|域偏移／负迁移|假设、混合分布、support和历史反例已连接|类先验估计与类别缺席、置信偏差及真实背景混淆分别解释|
|实验与响应解释|九种具体设计及其它有辨识力对照已展开|旧方法保留，旧packet预算不重写成通用限制；机械响应不等于任务收益|
|训练充分性／STOP／恢复|平台分类、延迟响应、日程、完整恢复与外推已建立|训练充分性和下一机会优先，无通用早停patience／最小轮次／跑满上限|
|选模／阈值／校准|已有具体方法、数学反例及部署身份说明|保持阈值拟合／最终评价／无标签代理的区别|
|经验迁移与更新|方法机制、真实／解释分离和正负空效应处理已展开|检索使用机制匹配与反例；旧数值不当新任务最优值|
|状态／执行／token|按需读取入口、实际正文打包和既有工具已接通|旧阶段要求标为历史，程序负责机械事实；不默认增加模型调用|

## 当前审查范围与边界

本轮完成了核心范围的内容与入口整合，并回查了关键数学／实现条件：加权归一化、新旧组件、D目标与时间尺度、各类先验估计、动态权重、日程和停止、排序／校准／阈值以及项目证据角色。旧有九类干预、实际反例和数据版本问题已保留；中文审阅与逐文件说明已生成。最终交付身份、来源与公开副本按完成审查记录核对。

知识建设不能声称穷尽全领域、达到人类专家或保证Agent不会误用。未实施的训练新方法保持方法知识身份；后续训练收益和行为验证不是本轮新增效果证据。覆盖表、引用和打包检查各自只证明对应范围。

本轮发现S005提取文本误写成知识索引，已保留误写文件，从原始HTML精确恢复；恢复后SHA-256与原文绑定一致，并重新读核相关优化器段落。误放的三份知识来源登记也恢复到正确索引。记录见control/S005_SOURCE_TEXT_RECOVERY_20261019.json。

当前成果是专业内容建设和可用性整合，不是新的模型效果证据。未启动训练、科学模型调用或公开验证，也未替换安装版。
