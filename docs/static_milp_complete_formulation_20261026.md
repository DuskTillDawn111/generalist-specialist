# 当前静态部署—路由联合优化的完整数学建模

本稿对照 `gsdeploy/milp.py`、`prediction.py`、`calibration_v2.py`、`surrogate_bounds.py` 和冻结的 `search-v2-v5-K8/system-config.json` 整理。对象为七模型静态预测域；不是在线队列感知策略的统一联合优化模型。帽号表示预测值。

完整问题采用六项非线性/离散回放评分器。实现以 MILP 主问题、完整预测器和离散身份反馈迭代求解，不声称已将 FIFO、P95 与全部资源函数一次性线性化。没有运行新实验或改变评分合同。

## 1. 集合、参数与决策变量

任务集 \(\mathcal T=\{\mathrm{text,image,OCR,ASR}\}\)，模型集 \(\mathcal M=\mathcal G\cup\mathcal S\)，三个通用、四个专用。\(\mathcal N_m\) 为模型 m 的冻结可选副本档位，包含 0。\(\lambda_t>0\) 是任务到达率，\(\alpha_t=\lambda_t/\sum_u\lambda_u\)。\(a_{tm}\in\{0,1\}\) 表示模型—任务支持关系；它不是绝对质量准入。

\(B,H,D,A,P_{ref},b_t\) 分别为显存预算、测量窗口、排空上限、启动摊销时长、功率参考值、各任务延迟参考尺度。冻结请求序列为

\[
\mathcal A_\xi=\{(a_i,t_i,u_i)\}_{i=1}^{N},\quad 0\le a_i<H,
\tag{1}
\]

其中 a_i 是计划到达时刻，u_i 是输入身份。输入顺序与候选路由无关；所有候选用同一序列。

\[
z_{mn}\in\{0,1\},\quad n_m=\sum_{n\in\mathcal N_m}nz_{mn},\quad
p_{tm}\ge0,\quad r_{tm}=\lambda_t p_{tm}.
\tag{2}
\]

z 选择副本档位；p 是模型级固定路由比例；r 是请求率。固定比例经平滑加权轮询映射到逐请求目的副本，不按理论比例直接近似有限样本分配。

## 2. 部署与有限路由域

\[
\sum_{n\in\mathcal N_m}z_{mn}=1\quad(\forall m),
\qquad
\sum_m p_{tm}=1,\quad 0\le p_{tm}\le a_{tm}(1-z_{m0})\quad(\forall t,m).
\tag{3}
\]

\[
\boldsymbol p_t\in\mathcal P_{t,K}:=
\bigcup_{d=1}^{K}\left\{\left(\frac{k_m}{d}\right)_m:
k_m\in\mathbb Z_{\ge0},\ \sum_m k_m=d\right\}.
\tag{4}
\]

式 (3) 筛除未部署与不支持路径；式 (4) 使用共同分母 1..K 的网格并集。K 是最大分母，不是模型个数。纯通、纯专和混合自然属于同一自由域，不强制任何结构。

冻结七模型主实验 K=8，没有容量见证扩展。通用代码支持额外两模型细分母网格/容量见证列；若启用，应显式改变候选域，并让 DFS 使用完全相同的列域。

## 3. 完整 FIFO 预测器

令 \(v=(\boldsymbol n,\boldsymbol p)\)，\(q_i(v)\) 为同一运行时 Router 得到的目的副本，\(m_i\) 为其模型，\(\pi(i;v)\) 为同副本前序请求。每任务模型流量均分到其驻留副本，名义每副本流量是 r_tm/n_m。

冻结逐输入 worker 服务值为 \(\widehat s^{worker}_{m,t,n,u}\) ms（单独画像的逐输入中位数乘开发集校准倍率）。令 g_m(n) 为配置的容量因子，预测有效服务时长为

\[
\widehat s_i(v)=\frac{\widehat s^{worker}_{m_i,t_i,n_{m_i},u_i}}{1000}
\frac{n_{m_i}}{g_{m_i}(n_{m_i})}.
\tag{5}
\]

\[
\widehat F_i=\max\{a_i,\widehat F_{\pi(i;v)}\}+\widehat s_i(v),\qquad
\widehat d_i=1000(\widehat F_i-a_i),
\tag{6}
\]

空队列前序完成时刻为 0；F/a 单位秒，d 单位 ms。g_m(0)=0，但未部署模型不进入上述分式。主实验对正副本档位配置 g_m(n)=n；这只是冻结代理假设，不是硬件多副本线性扩展证明。当前静态 v2 不按 kernel 重叠动态重估服务。

## 4. 六项总效用

\[
\widehat U(v)=-\widehat C(v),\qquad
\widehat C(v)=w_Q\widehat Q+w_T\widehat T+w_L\widehat L+w_P\widehat P+w_M\widehat M+w_E\widehat E,
\quad w_j\ge0,\quad\sum_jw_j=1.
\tag{7}
\]

**质量。** 设 I_t 为任务 t 的请求集合，N_t=|I_t|。按实际路由采用的冻结输出计算

\[
\widehat Q(v)=\sum_t\alpha_t\widehat q_t(v),\qquad
\widehat q_t=\begin{cases}
1-\frac{1}{N_t}\sum_{i\in I_t}\mathrm{EM}(\widehat y_i,y_i),&t=\mathrm{text},\\
1-\frac{1}{N_t}\sum_{i\in I_t}\mathbf1(\widehat y_i=y_i),&t=\mathrm{image},\\
\frac{\sum_{i\in I_t}\mathrm{EditDistance}(\widehat y_i,y_i)}{\sum_{i\in I_t}|y_i|_{char}},&t=\mathrm{OCR},\\
\frac{\sum_{i\in I_t}\mathrm{EditDistance}(\widehat y_i,y_i)}{\sum_{i\in I_t}|y_i|_{word}},&t=\mathrm{ASR}.
\end{cases}
\tag{8}
\]

CER/WER 为总编辑错误与总参考长度之比，不按单请求错误率简单平均，也不截断至 1。不设置绝对质量门槛。

**窗内完成损失与延迟。**

\[
\widehat T(v)=1-\frac{\sum_{i=1}^{N}\mathbf1(\widehat F_i\le H)}{N},\qquad
\widehat L(v)=\sum_t\alpha_t\frac{\frac1{N_t}\sum_{i\in I_t}\widehat d_i}{b_t},\qquad
\widehat P(v)=\sum_t\alpha_t\frac{\operatorname{P95}\{\widehat d_i:i\in I_t\}}{b_t}.
\tag{9}
\]

P95 是各任务经验分位数的加权值，不是所有任务混合后的 P95；与实现一致使用线性插值分位数。窗口后、排空内完成的请求仍进入 Q/L/P，但使 T 增加。

**显存。** 令 mem_m(n) 为冻结副本档位显存表（mem_m(0)=0），f 为安全系数，kappa_M>=1 为校准倍率，M_dev(n) 为已知相同部署的开发测量峰值（无匹配记录则该项为 0）：

\[
\widehat M_{safe}(v)=\max\left\{\kappa_M f\sum_m\mathrm{mem}_m(n_m),\ M_{dev}(\boldsymbol n)\right\},
\qquad \widehat M(v)=\widehat M_{safe}(v)/B.
\tag{10}
\]

**整卡能耗。** 记 s_base_mt(n) 为原 aggregate baseline（区别于逐输入延迟回放值与容量池均值），冻结非负功率系数为 beta_0,beta_b,beta_n，单副本启动能量系数为 j_0：

\[
\widehat W(v)=\beta_0+\beta_b\sum_{t,m:n_m>0}\frac{r_{tm}s^{base}_{mt}(n_m)}{1000}
+\beta_n\sum_m n_m,
\qquad
\widehat E(v)=\frac{\widehat W(v)+j_0\sum_m n_m/A}{P_{ref}}.
\tag{11}
\]

它对应真实评价定义 E=(J_run/H+J_once/A)/P_ref 的预测版本；整卡能量只算一次。主 E 不计排空阶段能量，排空能量另报诊断。

## 5. 完整静态优化问题

容量服务均值 s_pool_mt(n) 是逐输入中位服务值乘校准倍率后，对完整冻结输入池等权平均；不同于上式 s_base。必要容量约束为

\[
\widehat\rho_m(v)=\frac{\sum_t r_{tm}\overline s^{pool}_{mt}(n_m)}{1000g_m(n_m)}\le\rho_{max}=0.8\quad(n_m>0).
\tag{12}
\]

\[
\begin{aligned}
\mathrm{P1}:\quad \min_{\boldsymbol z,\boldsymbol p}\quad &\widehat C(v)\\
\mathrm{s.t.}\quad &(3),(4),\\
&\widehat M_{safe}(v)\le B,\\
&\widehat\rho_m(v)\le0.8\quad\forall m:n_m>0,\\
&\widehat F_i(v)\le H+D\quad\forall i,\\
&\text{所需性能、质量、显存、能耗证据完整，且每个正需求任务有观测。}
\end{aligned}
\tag{13}
\]

这里不约束 T=0。rho 筛查和有限队列排空不是实测稳定性证明；实测仍单独执行已有 steady/recoverable_congested 等分类。未知证据使候选未决，不能补零或当作物理不可行。

## 6. MILP 列变量与主问题

对每个任务 t，预枚举有限路由—副本列集合 C_t。每列 c 记录：路由率 r_tmc、被用模型的正副本状态 n_tmc、相应容量系数 h_tmc 和有效成本下界 b_tc。定义

\[
y_{tc}\in\{0,1\},\quad \sum_{c\in\mathcal C_t}y_{tc}=1,
\qquad r_{tm}=\sum_{c\in\mathcal C_t}r_{tmc}y_{tc},
\qquad
h_{tmc}=\begin{cases}\frac{r_{tmc}\overline s^{pool}_{mt}(n_{tmc})}{1000g_m(n_{tmc})},&r_{tmc}>0,\\0,&\text{其他}.
\end{cases}
\tag{14}
\]

副本关联与容量约束按当前代码的聚合形式为

\[
\sum_{c\in\mathcal C_t:r_{tmc}>0,\ n_{tmc}=n}y_{tc}\le z_{mn}
\quad\forall t,m,n>0,
\qquad
\sum_t\sum_{c\in\mathcal C_t}h_{tmc}y_{tc}\le0.8\sum_{n>0}z_{mn}\quad\forall m.
\tag{15}
\]

允许空闲驻留模型；不把“有部署必有流量”作为额外约束。令 m_mn^LB=kappa_M f mem_m(n)，状态下界系数为

\[
b^{state}_{mn}=w_M\frac{m^{LB}_{mn}}{B}
+w_E\frac{n(\beta_n+j_0/A)}{P_{ref}}.
\tag{16}
\]

若冻结 v2 的任务请求数 <=1024，保留该任务及其固定输入/Router 序列，删除所有其他任务，作单任务 FIFO 重放，得到 d_i^iso、N_late_tc^iso。则

\[
\begin{aligned}
b_{tc}={}&w_Q\alpha_t q_{tc}
+w_L\alpha_t\frac{\overline d^{iso}_{tc}}{b_t}
+w_P\alpha_t\frac{P95^{iso}_{tc}}{b_t}
+w_T\frac{N^{iso}_{late,tc}}{N}\\
&+\frac{w_E}{P_{ref}}\left(\alpha_t\beta_0
+\beta_b\sum_{m:r_{tmc}>0}\frac{r_{tmc}s^{base}_{mt}(n_{tmc})}{1000}\right).
\end{aligned}
\tag{17}
\]

固定服务、输入与副本分配后，删除其他任务只会提前该任务完成时刻，故均值、P95 和窗外数量均为下界。长任务轨迹退回逐请求无等待服务下界，T 下界为 0。未知路径的零仅用于下界松弛，完整评分仍为 UNKNOWN。上述服务下界不能无证明转用于动态 GPU 竞争预测器。

\[
\mathcal B(z,y)=\sum_{m,n}b^{state}_{mn}z_{mn}+\sum_t\sum_{c\in\mathcal C_t}b_{tc}y_{tc}
\le\widehat C(z,y).
\tag{18}
\]

令 theta>=0 为连续成本变量，第 k 轮主问题为

\[
\begin{aligned}
\mathrm{MP}_k:\quad\min_{z,y,\theta}\quad&\theta\\
\mathrm{s.t.}\quad&\sum_nz_{mn}=1\quad\forall m,\\
&\sum_cy_{tc}=1\quad\forall t,\\
&\text{式 (15) 的副本关联与容量约束},\\
&\sum_{m,n}m^{LB}_{mn}z_{mn}\le B,\\
&\theta\ge\mathcal B(z,y),\\
&\text{迄今生成的完整身份残差割、排除割和实验子域限制},\\
&z_{mn},y_{tc}\in\{0,1\},\ \theta\ge0.
\end{aligned}
\tag{19}
\]

主问题的显存约束只是必要下界筛查；式 (10) 的整部署校准显存、排空及完整六项仍由预测器校验。所有列/状态系数预计算后均为常数，因此 MP_k 是 MILP。

## 7. 完整评分与残差身份反馈

把全部 z/y 二进制变量合记 v，所有身份恰选 b=|M|+|T| 个变量。对第 j 个已完整评分的合法身份 v^j，令 I_j 为其取 1 的位置集合：

\[
h_j(v)=b-\sum_{\ell\in I_j}v_\ell,\qquad
C_j=\widehat C(v^j),\qquad\Delta_j=C_j-\mathcal B(v^j)\ge0.
\tag{20}
\]

添加

\[
\theta\ge\mathcal B(v)+\Delta_j[1-h_j(v)].
\tag{21}
\]

同一身份 h_j=0，右端等于 C_j；其他整数身份 h_j>=1，右端不高于原下界。因此该割在整个有限整数域有效，不依赖评分函数凸性。

若完整预测证实约束非法或超过排空上限，只排除该完整身份：

\[
\sum_{\ell\in I_j}v_\ell\le b-1.
\tag{22}
\]

UNKNOWN 身份也暂移出当前可评分主问题，但单独保留其身份、原因与成本下界；不把它解释为已证明不可行。代码对已评价身份的数值重复另作身份排除并保留 incumbent/未知界，不靠四舍五入制造证书。

## 8. 最优性界与终止

设 C_inc 为已知合法候选的最小完整成本，LB_MP 为剩余主问题的经求解器验证的成本下界，LB_unk 为所有暂移出 UNKNOWN 身份的下界最小值（空集时 +infinity）。包括已评估候选的安全全域界可写为

\[
LB=\min\{LB_{MP},LB_{unk},C_{inc}\},\quad
LB\le C^*\le C_{inc},\qquad
U_{inc}=-C_{inc}\le U^*\le-LB.
\tag{23}
\]

\[
\mathrm{gap}=C_{inc}-LB=(-LB)-U_{inc}.
\tag{24}
\]

完整建立冻结有限域、gap<=epsilon 且满足证据条件时才能发证；当前实现对发现的 UNKNOWN 采取保守口径，只要仍有记录就不返回正式最优证书。预算耗尽返回 incumbent/gap/状态。该证书不证明连续路由最优，也不证明所有真实 GPU 物理部署的最优。

## 9. 强竞争部署与 DFS 的统一数学定义

自由赢家为 v*=(n*,p*)，可行有限域为 F_K。最强其他部署、同部署最强其他路由分别由

\[
v_{dep}=\arg\min_{v\in\mathcal F_K:\boldsymbol n\ne\boldsymbol n^*}\widehat C(v),
\qquad
v_{route}=\arg\min_{v\in\mathcal F_K:\boldsymbol n=\boldsymbol n^*,\boldsymbol p\ne\boldsymbol p^*}\widehat C(v)
\tag{25}
\]

给出。排除赢家部署的线性式为 sum_m z_{m,n_m*}<=|M|-1；同部署其他路由先固定赢家副本档位，再排除赢家完整身份。最优不分流限制每个任务仅选支持集大小为 1 的列；纯专限制所有通用 n_m=0；纯通限制所有专用 n_m=0。受限问题若预算中止只能称候选。

同域 DFS 与 MILP 应满足

\[
|\widehat U^*_{MILP}-\widehat U^*_{DFS}|\le\epsilon,
\tag{26}
\]

并检查并列最优、域和评分合同。已完成的效率对照为五模型、0/1 副本、K=4 子域，不是七模型 K=8 全域 DFS 证书。GPU 配对独立验证有限候选表现，不把少数实测对照写成全部物理空间最优证明。

## 10. 当前冻结主实验参数

|参数|取值|
|---|---|
|模型|三个通用、四个专用|
|lambda_t|四任务各 0.0748360842705697 req/s|
|K|8|
|H / D|1800 / 300 s|
|B|49140 MiB|
|b_text,b_image,b_OCR,b_ASR|1000,250,10000,3000 ms|
|六项权重|各 1/6|
|P_ref / A|450 W / 3600 s|
|规划 / 确认 seed|42 / 43,44,45|
|容量见证|关闭|
|主目标|nominal 固定轨迹；非跨 seed 鲁棒目标|

正副本状态：G1={1,2,3,4}；G2={1,2,3,4,5}；G3={1,2,3,4}；text/image/ASR 各 {1,2,3,4}；OCR={1,2,3,4,6,8}。每个集合另加 0。

在线队列感知分流的动作依赖时间和可见状态，需引入策略 pi 及对应动态评价器。本稿不把它伪装成上述固定比例静态 P1 中已联合验证的决策变量。
