# 本轮验收记录

基线 `f68407f7bb5942e48018201f11da6fff7617a603`。测试在私人目录、已有 Python
3.12 / PyTorch 2.7.0 / Accelerate 1.12.0 环境完成；没有安装或升级包，没有
修改公共源数据。下列路径是本轮产物，不作为默认正式训练路径。

## 状态与框架

- `test_umi_training_state.py`：4 项通过。
- `test_umi_checkpoint.py`：5 项通过，包含保存中途故障注入、等长文件损坏、
  数据/统计/计划/卡数变更拒绝。
- 原有 sampler 回归：13 项通过；normalization Dataset/factory 回归：4 项通过。
- 单进程实际入口：`../runs/umi_training_final_1proc_v2/acceptance.json`。
- 双 CPU 进程实际入口：`../runs/umi_training_final_2proc/acceptance.json`。

两种实际入口对照均使用12次更新、每进程2个worker、预取和G=2。连续运行与
第3、6、9次更新后分别退出/重启的运行相比，模型、Adam、scheduler、每rank
RNG逐位一致；逐更新样本与学习率严格一致，并独立核对了全局 sampler 顺序。
尾部没有补重复样本，两个阶段均经过了epoch边界。另有5类入口拒绝检查通过。
每次计划暂停后，还故意向trace和主日志追加半行JSON；重启修复后继续比较，
结果仍逐位一致。恢复日志使用流式扫描，不把整场训练的日志装入内存。

CPU分布式明确使用Gloo。镜像中MPI可用，任由框架自动选择会使torchrun的两个
进程各自进入不匹配的MPI上下文；入口增加后端选择与WORLD_SIZE核对。
Accelerate通用unwrap会导入已安装的DeepSpeed并触发CPU测试不需要的Triton
驱动初始化，因此当前支持范围内使用只处理plain/DDP的unwrap，不修改环境。

## 真实 QwenPI

结果：**通过**。目录：`../runs/umi_staged_qwenpi_engineering_v2/`，摘要文件
`real_model_acceptance.json`，原始更新日志、样本trace与完整checkpoint均保留。

- 实际运行 Qwen3-VL-2B-Instruct + QwenPI，单PPU、G=2、worker=2。
- A/B工程视图分别1907/1103个窗口，各2次更新；共4次更新、8次窗口曝光。
- 第1次更新后完整保存并退出，新进程加载后完成2/3/4；阶段顺序A、A、B、B。
- 第1/2/4步checkpoint均完整发布。1025个已建立的Adam参数状态，其step分别
  全为1/2/4；scheduler的last_epoch也分别为1/2/4，没有阶段重置。
- 保存的实际样本顺序独立核对全局sampler通过。VLM、state encoder和动作头
  的抽查参数在第1～4步间均发生非零且有限的变化。
- 第2/4步执行独立工程评测。整个流程峰值张量显存 **62.23 GiB**；第一进程
  只更新一次的峰值为56.18 GiB，不能用它代表恢复后的完整流程峰值。
- loss依次为9426.34、26499.80、5626.75、4465.97。这是随机动作头的短程功能
  验收，不据此声称收敛、动作质量或泛化已验证。

真实模型验收时的代码快照保存在上述目录的`tested_code.patch`，
`tested_code_identity.json`已核对其完整starVLA源码摘要与run identity一致。
其源码SHA-256为`3429ffc04f9b1c16907b48d93e1a8bbe636eb36dbda535ce897f070569ea82f0`。
之后增加的日志尾部修复、发布目录fsync由CPU故障注入及进程恢复测试覆盖；
独立CPU进程的后端选择修正也由最终单进程测试覆盖。没有将这些后续修改
冒充为已重新执行完整Qwen训练。严格resume绑定代码身份；要续跑旧工程快照，
应还原记录的基线与补丁，不能修改checkpoint身份来绕过校验。

## 边界

本轮不证明策略效果或泛化；工程评测与训练数据有重叠。没有测试真实QwenPI
双PPU DDP、多节点、DeepSpeed/ZeRO、FSDP、64卡性能或全量训练。正式语义划分
和formal归一化合同仍待确定。普通DDP的功能测试不能替代这些后端与规模验收。
