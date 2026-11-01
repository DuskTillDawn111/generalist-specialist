# 通专模型联合部署、任务路由与验证：数学模型和算法

本稿参照 J3O 论文第 III 节及 Algorithm 1–2 的写法组织：先给系统变量与约束，再给目标问题、求解过程和验证过程。符号与方法按本项目当前实现重建，不将 J3O 的客户端—边缘—云系统或其近似保证直接移植为本项目结论。

**当前需要分开描述三部分：**已经验证的静态 MILP 部署/固定比例路由；已经实现并进行过开发实验的共享 GPU 在线分流；尚待硬件验证的资源划分扩展。三者还不是已完成验证的统一联合求解器。

## 1. 系统模型与决策变量

任务集为 \(\mathcal T=\{\mathrm{QA,image,OCR,ASR}\}\)，模型池为 \(\mathcal M=\mathcal G\cup\mathcal S\)，含三个通用模型和四个专用模型。固定工作负载为 \(\boldsymbol\lambda=(\lambda_t)_{t\in\mathcal T}\)，\(\Lambda=\sum_t\lambda_t\)，\(\alpha_t=\lambda_t/\Lambda\)。所有比较共享硬件、输入池、到达过程、测量窗 \(H\)、排空上限 \(D\)、参考尺度和效用权重。

|符号|定义|
|---|---|
|\(\mathcal N_m\)|模型 \(m\) 有画像的副本档位，含 0|
|\(z_{mn}\in\{0,1\}\)|是否选模型 \(m\) 的副本档位 \(n\)|
|\(n_m\)|实际部署副本数|
|\(a_{tm}\in\{0,1\}\)|模型—任务路径是否已验证可执行；不是质量达标标记|
|\(r_{tm}\ge0\)|任务 \(t\) 路由给模型 \(m\) 的请求率|
|\(p_{tm}=r_{tm}/\lambda_t\)|固定分流比例|
|\(\mathcal A_\xi\)|种子 \(\xi\) 下冻结的到达与输入序列|
|\(B,b_t,P_{\rm ref},A\)|显存预算、任务延迟尺度、功率尺度、启动摊销时长|
|\(K\)|有理数路由网格的**最大分母**，不是接收模型数|

部署和路由约束写为

\[
\sum_{n\in\mathcal N_m}z_{mn}=1,\quad
n_m=\sum_{n\in\mathcal N_m}n z_{mn},\quad z_{mn}\in\{0,1\},\quad\forall m. \tag{1}
\]

\[
\sum_{m\in\mathcal M}r_{tm}=\lambda_t,\quad
0\le r_{tm}\le\lambda_t a_{tm}(1-z_{m0}),\quad\forall t,m. \tag{2}
\]

对 \(\lambda_t>0\)，实际枚举的比例属于

\[
\mathcal P_{t,K}=\bigcup_{d=1}^{K}
\left\{(k_m/d)_{m\in\mathcal M}: k_m\in\mathbb Z_{\ge0},\ \sum_m k_m=d\right\},
\quad p_{tm}=0\text{ 若 }a_{tm}=0\text{ 或 }n_m=0. \tag{3}
\]

固定 \(n_m>0\) 时模型流量均分到其副本，\(r_{t,m,i}=r_{tm}/n_m\)，由与运行时相同的平滑加权轮询器映射到逐请求副本。因此有限样本实际分流由路由重放确定，不强行等于理论比例。若启用额外容量可行流量见证，须将其显式加入冻结域；公平 DFS 对照使用相同点集，不能双方各自产生不同见证。

纯通、纯专、混合都是 (1)–(3) 的可行子集。自由问题没有必须保留通用、必须混合或必须分流的约束。

## 2. 六项效用与资源约束

统一采用成本向量 \(\boldsymbol\ell=(Q,T,L,P,M,E)\)，越小越好：

\[
U(c;\xi)=-C(c;\xi)=-\sum_{j\in\{Q,T,L,P,M,E\}}w_j\ell_j(c;\xi),
\quad w_j\ge0,\quad\sum_jw_j=1. \tag{4}
\]

当前正式六项等权 \(w_j=1/6\)。这里 \(c=(\boldsymbol n,\boldsymbol r)\)。负的 U 是正常的成本记号；正分流收益指 \(\Delta U>0\)，不是要求 U 自身变正。

### 2.1 质量损失

\[
q_t(c)=\begin{cases}
1-\mathrm{EM}_t(c),&t=\mathrm{QA},\\
1-\mathrm{Acc}_t(c),&t=\mathrm{image},\\
\frac{\sum_i\mathrm{EditErr}_{ti}(c)}{\sum_i\mathrm{RefChars}_{ti}},&t=\mathrm{OCR},\\
\frac{\sum_i\mathrm{EditErr}_{ti}(c)}{\sum_i\mathrm{RefWords}_{ti}},&t=\mathrm{ASR},
\end{cases}\qquad Q(c)=\sum_t\alpha_tq_t(c). \tag{5}
\]

分流后按每条请求实际采用模型的输出计算；预测回放冻结的逐输入输出记录，GPU 实测重新评分当次输出。QA 的 F1 可以另报，但当前 Q 用 EM。CER/WER 不截断且可超过 1。**不存在绝对精度准入约束。**在线策略仅能使用历史聚合质量预测，不读取当前参考答案。

### 2.2 吞吐损失、平均延迟、P95

对完整注册的到达 cohort，令 \(N_{\rm off}\) 为计划请求总数，\(N_{\rm win}\) 为窗内完成数，\(d_{ti}\) 为从计划到达到主机收到结果的端到端时延：

\[
T(c)=1-\frac{N_{\rm win}(c)}{N_{\rm off}},\qquad
L(c)=\sum_t\alpha_t\frac{\overline d_t(c)}{b_t},\qquad
P(c)=\sum_t\alpha_t\frac{\operatorname{P95}(\{d_{ti}(c)\})}{b_t}. \tag{6}
\]

P 是各任务 P95 的流量加权，非混合全部任务后的全局 P95。排空完成请求仍进入 L/P/Q；吞吐分母仍是完整计划。当前 \(b_{QA}=1000\)、\(b_{image}=250\)、\(b_{OCR}=10000\)、\(b_{ASR}=3000\) ms。这些是冻结研究尺度，不是业务 SLO。

### 2.3 显存与能耗

\[
M(c)=\frac{M_{\rm safe}(c)}{B},\qquad M_{\rm safe}(c)\le B,\qquad B=49140\ \mathrm{MiB}. \tag{7}
\]

基础规划的加法驻留代理为 \(f\sum_m\mathrm{mem}_{m,n_m}\)。实际静态 v2 搜索使用校准后的 \(\widehat M_{\rm safe}(c)\)：加法安全驻留乘已拟合倍率，与已知部署峰值取大者。它是预测，不与实测驻留量混称。实测效用使用安全修正后的预热后/测量后拥有进程驻留快照最大值；运行采样峰值另作硬预算检查。

\[
E(c)=\frac{J_{\rm run}(c)/H+J_{\rm once}(c)/A}{P_{\rm ref}},\quad
J_{\rm run}=\int_{0}^{H}W_{\rm board}(\tau)\,d\tau,\quad
P_{\rm ref}=450\ \mathrm W,\ A=3600\ \mathrm s. \tag{8}
\]

整卡能量只积分一次，含板卡基础耗电；启动项含加载/priming/预热。当前主指标不含窗后排空电量，后者另报诊断。不能把各模型独占耗电相加替代同卡整体能量。

### 2.4 容量和正式有效性

静态校准 v2 的必要容量检查为

\[
\rho_m(c)=\frac{\sum_t r_{tm}\,\overline s^{\rm pool}_{mt,n_m}}{1000n_m}\le0.8,
\quad n_m>0, \tag{9}
\]

其中服务单位为 ms，池均值来自该副本档位的冻结完整输入池。一般独立代理采用其显式 capacity factor；不能假设真实多副本容量线性增长。\(\rho\) 只是规划约束，不等价于硬件排队稳定性证明。

正式实测的有效性另写为

\[
\mathsf V(c)=\mathsf{NativeOK}\land\mathsf{CompleteCohort}\land
\mathsf{MemoryOK}\land\mathsf{EvidenceComplete}\land
\{\mathsf{Stability}\in\{\mathrm{steady,recoverable\_congested}\}\}. \tag{10}
\]

复用现有稳定性分类，不在本稿重定义恢复阈值。缺质量/能耗/任务观测为 UNKNOWN，不按零补齐；未完成、非法和证据不足分别记录。预测器约束检查与真实稳定性门禁是不同层，不能把有限 FIFO 排空当成完整真实稳定性判据。

## 3. 问题 P1：静态部署与固定路由联合优化

用帽号区分冻结预测器与真实实测：

\[
\begin{aligned}
\mathrm{P1}:\quad&\min_{\boldsymbol z,\boldsymbol r}\ \widehat C_\xi(\boldsymbol z,\boldsymbol r)=-\widehat U_\xi(\boldsymbol z,\boldsymbol r)\\
&\mathrm{s.t.}\ (1)\text{--}(3),\quad\widehat M_{\rm safe}\le B,\quad\rho_m\le0.8,\\
&\qquad\text{预测 cohort 在有界排空内完成，所有需求任务均有观测，证据完整。}
\end{aligned} \tag{11}
\]

该问题的域记为 \(\mathcal D_K\)。当前 nominal 模式固定一份种子轨迹；robust 模式另定义服务压力情景并优化最坏情景。不能把没有执行的跨种子鲁棒优化写成当前默认目标。

### 3.1 完整候选评分器

冻结轨迹 \(\mathcal A_\xi=\{(a_i,t_i,u_i)\}_{i=1}^{N_{\rm off}}\)，其中 \(u_i\) 是输入标识。给定 c，通过路由器得到副本 \(q_i\)，按同副本 FIFO 计算

\[
S_i=\max(a_i,F_{\operatorname{prev}(q_i)}),\qquad
F_i=S_i+\widehat s_i(c),\qquad
\widehat d_i=F_i-a_i. \tag{12}
\]

时间单位统一。当前已验证静态 v2 的 \(\widehat s_i\) 是冻结逐输入、模型、任务、副本档位的中位服务时间乘开发校准倍率及既有容量转换。它不逐事件重估所有活动 kernel 的竞争，因此**不能把这个版本描述成已经实现 Roomie 式 kernel 序列预测**。整方案共驻画像是另一个显式性能模式，是否可用由画像身份决定。

### 3.2 列表示与主问题

每个任务列 \(c\in\mathcal C_t\) 保存路由比例、接收模型所需副本状态和有效成本下界；\(y_{tc}\in\{0,1\}\)。

\[
\sum_{c\in\mathcal C_t}y_{tc}=1,\qquad
y_{tc}\le z_{m,n_{tc,m}}\quad\text{对列内所有正流量模型 }m. \tag{13}
\]

由各列恢复 \(r_{tm}=\sum_c y_{tc}r^{tc}_{tm}\)。未被路由使用的模型仍允许驻留，因此额外驻留成本不会被漏掉。

令 \(v=(z,y)\)，构造全域有效下界 \(\mathcal B(v)\le\widehat C(v)\)：

\[
\mathrm{MP}:\quad\min_{z,y,\theta}\theta,\quad
\theta\ge\mathcal B(z,y)=b_0+\sum_{m,n}b^{\rm state}_{mn}z_{mn}
+\sum_{t,c}b^{\rm route}_{tc}y_{tc}, \tag{14}
\]

并满足 (1)、(13) 以及可线性化的显存/容量必要约束。v5 界包含精确冻结质量回放、有效显存/能耗分解以及单任务 FIFO 延迟/窗外完成下界。删除其他任务只能提前相同服务序列的 FIFO 完成，故该界有效。长轨迹退回逐请求无等待服务下界。上述证明只在对应冻结静态预测器下成立；动态竞争或新资源域必须重新证明。

### 3.3 残差身份反馈与停止

完整评分得到候选 \(v^j\) 的 \(C_j\)。令 \(I_j\) 为该身份选中的二进制变量集合，\(b=|I_j|\)，\(h_j(v)=b-\sum_{i\in I_j}v_i\)，\(\Delta_j=C_j-\mathcal B(v^j)\ge0\)。当前实现使用残差反馈：

\[
\theta\ge\mathcal B(v)+\Delta_j\,[1-h_j(v)]. \tag{15}
\]

当身份相同，右端为真实预测成本 \(C_j\)；其他完整身份 \(h_j\ge1\)，此约束不强于原下界。它不是对黑盒非凸函数假定凸性后加切平面。非法候选仅排除该身份：

\[
\sum_{i\in I_j}v_i\le b-1. \tag{16}
\]

证据未知时暂移出可评分主问题，同时保留其乐观成本界；不能称物理不可行。若 incumbent 成本为 \(C_{\rm best}\)，剩余主问题和未知集合的全域成本下界为 \(LB\)，则

\[
LB\le C^*\le C_{\rm best},\quad
U_{\rm best}\le U^*\le-LB,\quad
\mathrm{gap}=C_{\rm best}-LB. \tag{17}
\]

完整域建立、求解和证据检查均完成，且在容差内消除未决最优可能性时，才能给冻结有限域证书。限时返回 incumbent/gap/UNKNOWN；该证书不是连续比例或真实 GPU 全组合全局最优。

## 4. Algorithm 1：联合部署与固定路由搜索

```text
Input : 冻结模型/副本池、负载 λ、轨迹 Aξ、六项评价器、网格 K、容差 ε、预算
Output: 最佳 (n*, r*)、Û*、有限域证书或未决 gap、未知候选
 1  建立同一有限域 DK；按任务生成路由/副本列
 2  验证结构界 B；构造主问题 MP；Cbest ← +∞；Unknown ← ∅
 3  while 时间、建列和反馈预算均未耗尽 do
 4      求解 MP；取得经求解器验证的下界 LB 及整数候选 v
 5      if 已可证明 gap ≤ ε 且无影响证书的未决项 then return CERTIFIED
 6      if MP 无剩余候选 then break
 7      解码 v → (n,r)；在固定轨迹上执行完整 FIFO/质量/资源预测
 8      if 预测证据未知 then
 9          保留该身份及其下界至 Unknown；从当前主问题移出
10      else if 约束不合法或排空失败 then 添加身份排除式 (16)
11      else
12          计算六项成本 Cj；必要时更新 Cbest 和 incumbent
13          检验 B(v) ≤ Cj；加入残差反馈式 (15)
14  汇总主问题、已评分候选与 Unknown 的界
15  return incumbent、gap、完成/预算/未知状态
```

## 5. 在线分流：共享 GPU 外部代价的条件决策

本节形式化已存在的 `CoupledUtilityPolicy`，不是把它包装成 P1 的已验证联合扩展。部署固定，当前请求 i 的可见状态为 \(\mathcal H_\tau\)：历史完成、当前请求、全部驻留副本的 host 队列及已用时间代理。不含未来请求、真值答案或尚未完成请求的真实服务时长。

### 5.1 同时重估全部活动请求

活动队首集合 \(\mathcal A(\tau)\)，情景 \(\omega\in\Omega=\{\mathrm{mean,p90}\}\)。令 \(u_j(\tau)\) 为剩余归一化工作量，\(\widehat s_j(\mathcal A,\omega)\) 为当前活动状态下完成一单位工作的预测时长。事件更新为

\[
\delta=\min_{j\in\mathcal A}\{u_j\widehat s_j(\mathcal A,\omega)\},\qquad
u_j(\tau+\delta)=\max\left(0,u_j(\tau)-\frac{\delta}{\widehat s_j(\mathcal A,\omega)}\right). \tag{18}
\]

完成者出队、后继入队，然后**对全部活动路径重新估计服务与整卡功率**。剩余能量为各事件段 \(\sum_k\widehat W(\mathcal A_k,\omega)\delta_k\)，截至共同测量截止时刻。这样通用接收一个分流请求引起的专用/背景减速会被纳入，而不是仅比较新请求自身时长。

已有队首用受支持的条件残余预测初始化；host 已用时间不是直接观测的 GPU kernel 执行进度。mean/p90 是条件情景，不是收益置信界。

### 5.2 可比的观察前缀与分流判据

每个候选目的副本 a 都评估同一集合：已完成 + 当前所有 pending + 当前请求 i。定义条件代价

\[
\widehat C^\omega_\tau(a)=w_Q\widehat Q^\omega_\tau(a)+w_T\widehat T^\omega_\tau(a)
+w_L\widehat L^\omega_\tau(a)+w_P\widehat P^\omega_\tau(a)+w_E\widehat E^\omega_{\tau,remaining}(a). \tag{19}
\]

以同一集合、同一截止时刻和任务权重重算均值/P95；在线 Q 使用冻结家族—任务聚合损失。已耗/启动电量为当前动作共同项，比较时消去；其绝对值并未被线上算法测得。驻留显存代理相同不代表真实工作空间差为零，物理 \(\Delta M\) 仍未知。

先选专用基线，再选稳健相对改善的通用：

\[
a_S=\arg\min_{a\in\mathcal S_i^{resident}}\max_{\omega\in\Omega}\widehat C^\omega_\tau(a),\qquad
\Delta_i(a)=\max_{\omega\in\Omega}\left[\widehat C^\omega_\tau(a)-\widehat C^\omega_\tau(a_S)\right]
+w_M\overline{\Delta M}. \tag{20}
\]

\[
a_i=\begin{cases}
\arg\min_{a\in\mathcal G_i^{resident},\,\Delta_i(a)<-\eta}\Delta_i(a),
&\text{支持域/预算/冷却/令牌门禁通过且该集合非空},\\
a_S\text{ 或确定性专用回退},&\text{其他情况}.
\end{cases} \tag{21}
\]

\(\eta\ge0\) 是冻结余量。若显存差上界未提供，当前实现的额外保守惩罚省略；这不等于已知物理 \(\Delta M=0\)，更不构成物理六项净收益保证。预测外状态、缺背景功率、超待处理请求预算时不截断队列，回退专用。完成反馈超预测漂移阈值触发冷却；没有用真值答案或在线重新拟合保证收益。

## 6. Algorithm 2：共享 GPU 条件效用分流

```text
Input : 当前请求 i、可见历史 Hτ、固定驻留、共享服务/功率提供者、六项权重
Output: 目的副本 ai、预测比较与回退原因
 1  处理已完成反馈；构造全部驻留副本 FIFO 快照
 2  核查支持域、令牌、漂移冷却与 pending 预测预算
 3  若预测预算超限，直接返回确定性专用回退（不裁剪 pending）
 4  for 每个可执行的驻留目的副本 a do
 5      将同一当前请求追加至 a 的队列副本
 6      for ω ∈ {mean, p90} do
 7          以式 (18) 重放完整可见 cohort；每次活动变化更新全部路径
 8          积分一次整卡剩余功耗；按式 (19) 计算条件代价
 9      任一所需证据缺失则将 a 标记 UNKNOWN
10  按式 (20) 选已知的最优专用基线 aS
11  按式 (21) 核查各通用的最坏情景相对改善
12  有通过门禁的严格改善则选其最优者，否则使用专用基线/回退
13  提交请求并记录候选、情景差值、活动事件、门禁和决策
```

式 (19)–(21) 优化观察前缀条件代理，而非完整未来窗口的真实 U。之前实测仍有负收益，因此算法表达应保留其适用范围，不把条件通过改写成稳定正分流已证明。

## 7. 验证：同域 DFS、强竞争方案与原生配对

### 7.1 DFS 与效率

\[
\widehat U_{\rm DFS}^*=\max_{c\in\mathcal D_K,\,c\text{ 预测可行}}\widehat U(c),\qquad
|\widehat U_{\rm MILP}^*-\widehat U_{\rm DFS}^*|\le\varepsilon. \tag{22}
\]

并列最优允许不同部署/路由，核对效用和完整 ties。plain DFS 关闭效用界剪枝，但保留合法显存/容量必要筛查；bounded DFS 使用经证明的同一结构界。域、输入、评分、容差及每次新缓存公平一致。记录总墙钟、完整预测次数、状态数、峰值内存、首次可行/最优时间和终局 gap。

设 \(N=|\mathcal D_K|\)、评分成本为 \(T_{eval}\)，穷举最坏成本为 \(O(NT_{eval})\)；主算法若完成 I 轮，成本为建列开销加 \(\sum_{j=1}^{I}(T^{j}_{MILP}+T^{j}_{eval})\)。主问题为整数优化、候选域也可能指数增长，不宣称 J3O 的贪心近似界或多项式复杂度适用于本算法。

### 7.2 强竞争方案由额外优化定义

令自由赢家为 \(c^*=(n^*,r^*)\)，不是只和纯通/纯专标签比较：

\[
c_{dep}=\arg\max_{c\in\mathcal D_K:n\ne n^*}\widehat U(c),\quad
c_{route}=\arg\max_{c\in\mathcal D_K:n=n^*,\,r\ne r^*}\widehat U(c), \tag{23}
\]

\[
c_{nosplit}=\arg\max_{c\in\mathcal D_K:|\operatorname{supp}(r_t)|=1,\,\forall t}\widehat U(c). \tag{24}
\]

纯通/纯专另作为受限结构控制。预算中止只能写 budget candidate，不能称该类最优。重复身份去重；少量原生对照验证候选优势，不能凭几个次优实测无条件证明所有未测物理组合更差。

### 7.3 分流收益与配对验证

同驻留允许/禁止分流，对每个独立种子 \(\xi\)：

\[
\Delta U_\xi=U(c_{spill};\xi)-U(c_{no\ spill};\xi)
=\sum_j w_j\left[\ell_j(c_{no\ spill};\xi)-\ell_j(c_{spill};\xi)\right]. \tag{25}
\]

这样可同时说明减少排队/能耗的收益和通用变慢、质量损失、P95 或其他请求受影响的负收益。改变驻留或资源模式时须另分解，不把综合收益全部归于分流。按预冻结顺序运行同轨迹、同输入配对；开发选择与确认隔离，正式输出重新评分。证据未知不算 0，不从均值中悄悄删去。

## 8. Algorithm 3：完整验证流程

```text
Input : 冻结研究域、校准/留出/确认集合、搜索与 GPU 预算
Output: 有限预测域正确性、效率结果、原生配对证据与局限
 1  单模型/多副本画像与有限共驻校准；冻结指标/归一化/输入身份
 2  独立留出验证预测器；失败则保留失败并修订新版本
 3  冻结预测器、候选域和源码，运行 Algorithm 1
 4  同域独立运行 plain/bounded DFS；核对最优值、ties、gap
 5  交替顺序独立计时，报告完整评价调用与总时间
 6  通过式 (23)–(24) 生成强竞争方案，加入纯通/纯专控制并去重
 7  同驻留分流实验：Algorithm 2 对照禁止分流及旧策略
 8  使用匹配轨迹和输入串行执行原生配对；保留全部终点
 9  从当次输出/延迟/显存/整卡功率重算六项 U，并核查稳定性
10  按预声明门禁决定独立确认是否启动；未通过如实报告
11  分别报告预测域最优、CPU效率、硬件效用与分流机制
```

## 9. Roomie / USHER 资源扩展的数学位置

后续可研究 \(c=(n,r,h,\pi)\)：h 是经平台确认的执行资源配置，\(\pi\) 是可执行路由/准入策略。目标可写 \(\max_{c\in\mathcal D^{resource}}U(c)\)，但目前没有已完成的四者联合优化与物理最优证书。

真正互斥 SM 分配需满足 \(\mathcal S_a(h)\cap\mathcal S_b(h)=\varnothing\) 及平台容量/粒度约束；MPS percentage 只是上限，不能用 \(\sum_a p_a=100\%\) 代替互斥证明。host 整请求并发限制可以写为 \(\sum_{a\in G}I_a^{running}(\tau)\le L_G\)，还须将许可等待纳入预测和端到端时延。当前实现具备契约/准入软件，真实 SM 绑定、准入感知预测、kernel 序列模型及 GPU 效益仍待验证。故这些内容在图中标为后续扩展，不画成已执行主流程。

## 10. 实现依据与论文图

- 六项公式：`gsdeploy/prediction.py::components`、`quality.py`、`energy.py`、`evaluation.py`。
- 静态搜索：`gsdeploy/milp.py`、`optimizer.py`、`surrogate_bounds.py`；逐输入界：`docs/static_exact_input_bounds_v5_20261022.md`。
- 在线算法：`tools/queue_coupled_utility_20261023/{policy,model,resources}.py`。
- 资源控制边界：`docs/resource_control_cpu_implementation_20261024.md`。
- J3O 参考：附件 `Batching-Aware Joint Model Onloading and Offloading for Hierarchical Multi-Task Inference.pdf` 第 6 页 Algorithm 1–2。它使用交替优化、贪心拉格朗日上载和固定上载后的 LP 分流；本项目是有限域联合 MILP + 预测反馈，以及独立的条件分流策略，不能照搬其算法名称/保证。

图表与 LaTeX 伪代码见 `artifacts/j3o-style-method-20261026-v3/`。本次只做方法整理与绘图，没有运行新 GPU 实验或改变主线程算法。
