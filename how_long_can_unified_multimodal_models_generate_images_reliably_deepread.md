# How Long Can Unified Multimodal Models Generate Images Reliably? Taming Long-Horizon Interleaved Image Generation via Context Curation (UniLongGen)

> **作者:** Haoyu Chen, Qing Liu, Yuqian Zhou, He Zhang, Zhaowen Wang, Mengwei Ren, Jingjing Ren, Xiang Wang, Zhe Lin, Lei Zhu
> **单位:** 香港科技大学(广州)、Adobe Research、华中科技大学、香港科技大学
> **发表:** Preprint, arXiv:2603.07540, 2026 年 3 月
> **链接:** https://arxiv.org/abs/2603.07540
> **代码 / 权重 / 数据:** ❌ 均未在论文中给出开源链接(训练-free 方法,基座为 BAGEL)

---

## TL;DR(一句话总结)

UniLongGen 是一个**训练-free 的推理期上下文整理(context curation)策略**:它通过一次性注意力探针(one-shot attention probing)读取统一多模态模型(基座为 BAGEL)自身的注意力相关性信号,在生成每张新图前从 KV-cache 中**丢弃**(而非压缩)无关的历史视觉 token,从而在 40 图交错生成基准上把 HPS v3 从 dense-KV 的 **3.17 提到 7.57**、DINOv2 身份一致性从 **0.316 提到 0.427**,同时实现最高约 **11× 的推理加速**。

---

## 1. 研究背景与动机

### 1.1 问题定义

统一多模态模型(Unified Multimodal Models, UMMs)正从"描述一张图"走向"在单一自回归流里规划、推理、渲染":模型可以写一段文字、生成一张图、继续叙事、再生成下一张图,如此交错(interleaved)迭代几十轮。这能支撑**角色一致的连环画、长篇故事板、迭代式视觉设计**等应用。

但实践者反复观察到一个**可靠性悬崖**:随着序列变长,图像质量迅速崩塌——结构瓦解,身份/风格漂移,尽管模型在短上下文下表现很强。本文研究的正是这个**长程交错生成(long-horizon interleaved generation)**的崩溃机制。

### 1.2 为什么重要

如 Figure 2 所示,作者在 40 图序列上测量四个质量指标(Consistency、Quality、HPS v3、PickScore),发现所有指标在前 ~20 图保持稳定,随后在 20–25 图窗口内**急剧崩塌**,几步之内多数样本退化为不可识别的图像。这意味着当前 UMM 的"有效生成长度"被严重限制,长篇视觉创作落不了地。

![Figure 2: 质量随序列长度增加而退化](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig2_degradation.png)

*Figure 2:在 1024×576 分辨率下生成 40 图,四个归一化指标在前 ~20 图稳定,随后进入"Degradation Phase"急剧下滑。右侧示例图清晰展示早期 shot 高保真、后期 shot 严重伪影。*

### 1.3 既有工作的局限(被点名的)

传统观点把这种崩溃简单归因于**"长上下文问题"**(long-context problem)——即 token 溢出或显存约束。本文明确反驳:

- **不是 token 数量的问题**:模型可以无缝处理 100k 文本 token,但在不到 20 张图后就崩溃。瓶颈由**图像事件(image events)的数量**驱动,而非原始 token 数。
- **不是被动稀释(passive dilution)**:在纯文本主导的历史里,无关文本只是稀释注意力(导致欠条件化、模糊);但密集视觉历史会**主动污染(active pollution)**——历史视觉 key 与生成目标共享同一表示空间,会主动**操纵(steer)**像素级合成轨迹。
- **通用长上下文加速方法(token 压缩/稀疏注意力)不适用**:它们的目标是"塞进更多 token",却可能保留那些触发污染的高方差离群 key。

### 1.4 本文填补的空白

作者把长程崩溃重新诠释为一个 **Softmax 注意力预算下的零和分配问题(allocation problem)**,提出两个核心诊断洞见:

1. **Event Bottleneck(事件瓶颈)**:质量退化是图像事件数量的函数,而非原始 token 数。作者称之为"有效多模态上下文长度(Effective Multimodal Context Length)"——由 Softmax 能可靠分辨的离散视觉事件数量决定。
2. **Attention Competition under Dense Visual History(密集视觉历史下的注意力竞争)**:每张新图引入大量视觉 token,形成"重尾(heavy-tailed)"噪声分布——大量(常常无关的)历史视觉 key 成为活跃竞争者,偶发的高相似度离群值会被 Softmax 指数级放大,从而"劫持(hijack)"注意力,向当前合成步注入有害信号。

由此动机:长程生成应当**优先保证安全的条件化(safe conditioning),而非全量记忆(total recall)**——模型必须**主动整理(actively curate)**历史,抑制或删除竞争性视觉信号。

---

## 2. 相关工作

### 2.1 统一多模态模型(UMMs)

两条生成范式:

- **纯自回归(AR)**:把所有模态 token 化为单一序列做 next-token 预测。代表:[Chameleon (Team, 2024)](https://arxiv.org/abs/2405.09818) 早融合统一词表;[Emu3 (Wang et al., 2024)](https://arxiv.org/abs/2409.18869) 证明 next-token 预测就能很强。后续主要在视觉 token 化上变化:像素级码([Li et al., 2025](https://arxiv.org/abs/2509.03498))、语义 token([Lin et al., 2025](https://arxiv.org/abs/2506.03147);[Jiao et al., 2025](https://arxiv.org/abs/2503.20853);[Qu et al., 2025])、可学习 query 表示([Pan et al., 2025](https://arxiv.org/abs/2504.06256);[Chen et al., 2025a](https://arxiv.org/abs/2505.09568))。
- **混合 AR-扩散(Hybrid AR-diffusion)**:AR LLM 配扩散/流解码器。代表:[Transfusion (Zhou et al., 2024)](https://arxiv.org/abs/2408.11039) 联训 next-token + 扩散;[Show-o (Xie et al., 2024)](https://arxiv.org/abs/2408.12528);[JanusFlow (Ma et al., 2025)](https://arxiv.org/abs/2411.07975) 结合 AR 与 rectified flow;[LMFusion (Shi et al., 2024)](https://arxiv.org/abs/2412.15188) 改造预训练 LLM;[VINO (Chen et al., 2026)](https://arxiv.org/abs/2601.02358) 扩展到交错全模态上下文。其它:Hunyuan-Image 3.0、[NextFlow (Zhang et al., 2026)](https://arxiv.org/abs/2601.02204)、[OmniGen2 (Wu et al., 2025)](https://arxiv.org/abs/2506.18871)、[Uni-X (Hao et al., 2025)](https://arxiv.org/abs/2509.24365)。为缓解模态异质性,还有 MoE 风格设计如 [BAGEL (Deng et al., 2025)](https://arxiv.org/abs/2505.14683)、Mogao、[HBridge (Wang et al., 2025b)](https://arxiv.org/abs/2511.20520)。

### 2.2 交错图文生成(Interleaved Generation)

要求模型产出任意交替模态的连贯序列。[Emu2 (Sun et al., 2024)](https://arxiv.org/abs/2312.13286) 和 [SEED-X (Ge et al., 2024)](https://arxiv.org/abs/2404.14396) 在大规模交错语料(如 [OBELICS](https://arxiv.org/abs/2306.16527)、CoMM)上训练展现强 in-context 学习;[MiniGPT-5 (Zheng et al., 2023)](https://arxiv.org/abs/2310.02239) 引入 generative vokens;[DreamLLM (Dong et al., 2023)](https://arxiv.org/abs/2309.11499)、[Kosmos-G (Pan et al., 2023)](https://arxiv.org/abs/2310.02992) 关注协同理解-生成;[Orthus (Kou et al., 2024)](https://arxiv.org/abs/2412.00127) 用模态专属头。

### 2.3 定位

本文不提出新架构、不训练,而是**在推理期**用模型自身注意力信号做上下文整理。它与"长上下文加速"(token 压缩、稀疏注意力)有本质区别:后者旨在"塞进更多",本文旨在"删除有害",优先 **relevance-filtered sparsity**(相关性过滤的稀疏化)而非 total recall。

---

## 3. 核心方法

本文方法部分分两块:**§3-4 是诊断(机制分析)**,**§5 是解法(UniLongGen)**。我严格按论文叙事顺序展开。

### 3.1 诊断 I:问题设定与现象(§3.1)

**任务设定。** 长程交错生成是单一连续序列,文本段与离散图像-token 块交替生成,上下文随每张图追加而增长。作者用 $N=40$ 图(与 $T$ 个文本段交错),在多个随机种子上评测(i)逐图质量,(ii)跨图一致性,随图索引 $i \in \{1,...,N\}$ 变化。

**基座模型:BAGEL。** 这是一个**混合 AR-扩散**统一模型:

- AR LLM + 扩散式图像生成器,文本与图像在同一序列里表示。
- 关键:它**同时使用 ViT 特征(语义理解)和 VAE latents(图像生成)** 两套视觉表示。这一点与作者的分析高度契合——它把"语义 grounding"和"合成动态"分离开。

### 3.2 诊断 II:图像数量比 token 数更重要(Observation 1, §3.2)

作者在三种分辨率(1024×1024、1024×576、768×432)下跑长程生成,它们的 token 增长速率差异巨大。**关键发现**(Figure 3):无论总 token 数如何,质量都在大约 **20–25 图的同一语义窗口**内崩塌。

![Figure 3: 瓶颈取决于图像数量而非 token 长度](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig3_bottleneck.png)

*Figure 3:横轴为图像索引,纵轴为累积 token 数,颜色表示生成质量(HPS v3)。固定显存预算下:~150k token 时,17 图(1024×1024)仍高保真,但 30 图(1024×576)同样 token 数下已完全退化。水平虚线(相同 token 预算对比)证明:匹配 token 预算并不能预测成功,图像数量才是主导瓶颈。*

这排除了简单的算力上限解释:模型能处理 150k token 当它们代表更少图像,却在相同 token 数下因代表更多图像而失败。作者称之为 **Event Bottleneck(事件瓶颈)**:有效上下文长度受**视觉事件数量**限制,而非 token 数。这把优化目标从"塞进更多 token(压缩/架构)"转向"管理事件级竞争(整理/curation)"。

### 3.3 诊断 III:模态不对称失败(Observation 2, §3.3)

作者设计了一个**匹配总上下文长度**的受控对比:(A)120K token 的纯文本历史 vs(B)token 数匹配的 22 图密集图像历史。Figure 4 显示清晰的定性差异:

![Figure 4: 文本历史 vs 图像历史引发不同的注意力机制](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig4_attn_regimes.png)

*Figure 4:上排(诊断)——文本历史的相似度分数分布**高度集中**(consistently focused);图像历史则**高方差、重尾**(更多离群值)。下排(结果)——文本历史主要导致**被动稀释**(模糊、欠条件化),图像历史导致**主动污染**(伪影、speckle、结构扭曲、身份/风格漂移)。*

**核心论断:密集视觉历史的独特风险。** 文本主导历史 → 失败是被动的(标准长上下文检索效率问题,相关证据被欠权重);密集视觉历史 → 失败是**主动的、结构性的**——因为历史视觉 token 与目标图像共享同一表示空间,它们会主动操纵像素级合成轨迹。无关视觉特征不是被动遗漏,而是被混入条件化表示并跨步累积。

### 3.4 诊断 IV:注意力劫持机制(§4)

这是全文最深入的机制分析,分三个子层次。

#### (a) 视觉竞争与注意力预算崩溃(§4.1)

作者用**注意力分布的熵(entropy)**量化生成时的注意力聚焦程度。Figure 5 显示:随上下文增长(尤其当更多图像 token 出现),**注意力熵系统性上升**(正相关 r=0.866),说明模型越来越"困惑于该看哪里"。

进一步,作者追踪 **key-reference attention mass**(分配给已知相关历史图像 token 的注意力,按历史图像总注意力归一化)。Figure 7 显示:随干扰图累积,这个相关性份额**急剧下降**(在 "one-key-reference + N distractors" 受控实验中从 ~50% 掉到 ~4%,约 **95% 相对损失**)。

![Figure 7: 冗余图像历史下的注意力错配](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig7_misallocation.png)

*Figure 7:左——无关历史仍吸引注意力,只有少数组(Image 1,6,11,16)含角色相关参考,但大量无关图像获得非平凡注意力,稀释真正参考的预算。右——key-reference 注意力随干扰图增加而陡降(~95% 相对损失)。*

**结论:这是分配失败(allocation failure)而非信息失败(information failure)**——正确信号仍在上下文里,只是越来越难被分辨出来。

#### (b) 重尾劫持的主动污染(§4.2)

**现象:图像历史诱发重尾匹配 + top-heavy 注意力。** 单张图贡献成千上万 patch 级 key,即便语义无关也有非平凡概率产生高点积**离群值**(incidental alignment)。注意力本身又是 top-heavy 的(Figure 6:top-10% token 常捕获一半以上预算)。因此少数极端视觉匹配可主导注意力——这解释了为何退化由**事件**(§3.2)而非原始 token 数主导:每张新图都引入一池新鲜密集 key,增加遇到劫持预算的离群值的概率。

**Softmax 把尾部风险放大为劫持,把稀释变成污染。** 因为 Softmax 是指数放大器,一个虚假的高相似度参考不只是加背景噪声——它**劫持大量注意力预算**,锁定到无关高频 patch,从而覆盖原本的条件化(真实主体参考或 prompt 线索),直接向当前合成步注入错误纹理和结构伪影。

> **Implication 1:需要事件级整理来消除尾部风险。** 为"塞更多 token"设计的技术(token 压缩、通用稀疏注意力)对统一生成不够,它们可能恰好保留那些触发污染的高方差离群值。稳定性要求**删除竞争源**(基于相关性丢弃整个图像块),截断重尾,防止劫持。

#### (c) 深度与层级专业化(§4.3)

基于扩散/流的统一生成器在多步上精炼同一组图像 token。作者发现 Transformer 深度是**专业化的**(Figure 8):

![Figure 8: 注意力在深度上的模态专业化](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig8_depthspec.png)

*Figure 8:早期层分配更多注意力给 Text/ViT token(grounding 和多模态路由),后期层越来越优先 VAE token(驱动图像合成)。Special/control token 在各层保持稳定份额。*

Figure 9 用 Ward 层次聚类进一步揭示五个功能簇:C1(Layer 0)文本主导,C2-C3 平衡文本/视觉,C4-C5(后期层)VAE 主导且熵更高。这些连续功能块把网络分成**文本依赖的早期区**和**生成主导的后期区**。

> **Implication 2:层分割的相关性估计。** 单一相关性 mask 不够——必须用**文本相关性**为早期层打分(保证指令跟随),用 **VAE 相关性**为后期层打分(保证一致性)。这直接催生了 UniLongGen 的双探针设计。

### 3.5 解法:UniLongGen(§5)

UniLongGen 只在图像合成期生效,对每张新图执行**三步流程**(Figure 10):

![Figure 10: UniLongGen 总览](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig10_overview.png)

*Figure 10:(a)双探针结构——早期层用 Text filter,后期层用 VAE filter;(b)注意力探针与选择——以待生成图像的 noisy VAE token 为 query,对历史 KV-cache 做探针,选 top-K,丢弃非选中 token。*

#### 步骤一:一次性上下文探针(One-Pass Context Profiling)

用**完整历史(dense KV)** 做一次前向,探测模型内部相关性。

- **输入/输出:** 输入是完整交错历史 + 当前待生成图像的 VAE query token;输出是各历史块的相关性分数。
- **为什么只探一次:** 把相关性估计与生成解耦,从而在所有后续扩散/流步上维持稳定的上下文策略,避免逐步重新探针带来的不一致(§A.2 指出逐步重分配会破坏生成稳定性)。

#### 步骤二:双深度打分(Dual-Depth Scoring)— Attention-Based Relevance Scoring(§5.1)

**核心创新:Pre-Softmax 聚合相关性打分。** 为降低单头高频虚假匹配的噪声,作者用 query 与 key 的**头平均点积**(pre-softmax,不过 Softmax),并做长度归一化。

把交错历史记为 $m$ 个 turn,索引 $i \in \{1,...,m\}$。每个 turn $i$ 含文本段 $T_i$ 和 VAE 图像块 $V_i$。令 $V^{\text{cur}}$ 为当前待合成图像的 VAE token 集合。$\mathbf{q}_{v,h}^{(\ell)}, \mathbf{k}_{u,h}^{(\ell)} \in \mathbb{R}^d$ 表示第 $\ell$ 层、第 $h$ 个头的 query/key 切片。

**(公式 1)先把当前图像的生成意图压缩成单个均值 query 向量:**

$$\bar{\mathbf{q}}^{(\ell)} \triangleq \frac{1}{|V^{\text{cur}}|}\sum_{v \in V^{\text{cur}}} \mathbf{q}_v^{(\ell)}$$

*通俗解释:对当前图像所有空间 VAE token 的 query 求平均,得到一个代表"我现在想画什么"的单一意图向量。这避免了逐 token 噪声。*

**(公式 2)对任意历史块 $B$(文本或 VAE)定义相关性分数 $S(B;\ell)$ 为块内 key 与均值 query 的平均头平均点积:**

$$S(B;\ell) = \frac{1}{|B|}\sum_{u \in B}\underbrace{\left(\frac{1}{H\sqrt{d}}\sum_{h=1}^{H}\bar{\mathbf{q}}_h^{(\ell)} \cdot \mathbf{k}_{u,h}^{(\ell)}\right)}_{\text{头平均相似度}}$$

*通俗解释:对块 $B$ 里每个 token,算它与当前意图的相似度(对 $H$ 个头平均、按 $\sqrt{d}$ 缩放),再对整个块求平均。块级平均是关键——它把单个高频离群 patch 的影响摊薄,正面应对 §4.2 的重尾劫持。这就是为什么 Pre-Softmax 优于 Post-Softmax(见 §F.2 消融)。*

#### 步骤三:模型对齐的双探针(Model-Aligned Dual Probing,§5.2)

利用 §4.3 的层专业化,在两个深度应用打分函数:

- **文本选择(Grounding):** 在早期层 $\ell_{\text{grd}}$(默认 $\ell_{\text{grd}}=1$),选相关性 top-$K_{\text{grd}}$ 的文本块:
  $$\mathcal{K}_{\text{grd}} = \text{TopK}\big(\{S(T_i;\ell_{\text{grd}})\}_{i=1}^{m}, K_{\text{grd}}\big)$$
- **图像选择(Synthesis):** 在后期层 $\ell_{\text{syn}}$(默认 $\ell_{\text{syn}}=15$),选相关性 top-$K_{\text{img}}$ 的 VAE 块,保证视觉一致性:
  $$\mathcal{K}_{\text{syn}} = \text{TopK}\big(\{S(V_i;\ell_{\text{syn}})\}_{i=1}^{m}, K_{\text{img}}\big)$$

作者发现 **$K_{\text{img}} \approx 4$** 的小预算在"提供足够视觉参考"和"最小化注意力污染"间最优。

#### 步骤四:生成期 KV 驱逐(KV Eviction During Generation,§5.3)

一旦 $\mathcal{K}_{\text{grd}}$ 和 $\mathcal{K}_{\text{syn}}$ 确定,UniLongGen 在后续所有生成步强制一个**固定的层分割可见性策略**。关键设计:**严格驱逐(evict)非选中 token,而非压缩**——确保虚假视觉离群值被彻底移出 Softmax 竞争,且不引入压缩带来的分布偏移合成特征。

令 $\mathcal{H}^{(\ell)}$ 为第 $\ell$ 层可见历史,策略为(**公式 3**):

$$\mathcal{H}^{(\ell)} = \begin{cases} \mathcal{T}_{\text{text}}(\{1\} \cup \mathcal{K}_{\text{grd}}), & \ell < \ell_{\text{syn}} \\ \mathcal{T}_{\text{img}}(\{1\} \cup \mathcal{K}_{\text{syn}}), & \ell \geq \ell_{\text{syn}} \end{cases}$$

*通俗解释:早期层(< $\ell_{\text{syn}}$)只看选中的**文本块**做 grounding;后期层(≥ $\ell_{\text{syn}}$)只看选中的**VAE 图像块**做合成。被选 turn 永远包含 turn 1(锚点)加 top-$K$ turn。这与模型的功能层级对齐:早期层用相关文本描述 grounding 生成,后期层关注相关视觉参考做合成。*

### 3.6 直觉解释(Intuitive Explanation)

把长程生成想象成一个画家在画连环画的第 30 格。dense-KV 相当于**强迫他同时盯着前面 29 格的每一个像素**——其中绝大多数(背景、道具、无关镜头)和当前要画的主角无关,但因为它们"看起来像图像",画家的眼睛会不由自主被某些高对比的无关细节吸引(注意力劫持),结果把无关纹理画进了主角脸上(主动污染)。

UniLongGen 的做法是:**先快速扫一眼全部历史(一次性探针),问模型自己"画这一格,你真正需要参考哪几格?"**,然后**把其余的格子盖起来**(KV 驱逐),只留 ~4 张最相关的参考图 + 几段相关文字描述。而且,**对"理解剧情"用文字提示(早期层),对"画得像"用图像参考(后期层)**——因为模型大脑的不同区域(层)本就分工不同。盖起来而非缩小(丢弃而非压缩),是因为把无关格缩成缩略图反而可能保留那个最碍眼的高频离群点。

---

## 4. 数据构建

本文不构建训练数据(训练-free),但构建了一个评测基准 **Long-Horizon Narrative Benchmark**(§B.1)。

### 4.1 数据来源与规模

- **50 个精心策划的叙事序列(narrative templates)**,每个序列含预定义的连贯剧情进展和具体视觉描述。
- 每个叙事含 **40 个图像生成槽位($N=40$)**,即每次完整 benchmark run 生成 **2,000 张图**(50×40),构成非平凡的长程压力测试。

### 4.2 交错结构与任务形式

数据组织为交错序列(**公式 4**):
$$\mathcal{S} = \{T_1, I_1, T_2, I_2, ..., T_N, I_N\}$$
其中 $T_i, I_i$ 分别是第 $i$ 个文本和图像分量。

采用 **fixed-text 协议**:文本 $\{T_i\}$ 逐字提供作为条件,模型只负责图像生成。目标是给定累积上下文 $\mathcal{C}_i = \{T_1, I_1, ..., T_i\}$ 生成图像 $I_i$(**公式 5**)。这隔离了文本生成方差,聚焦评估视觉一致性和上下文遵循。

### 4.3 主体复现策略(Subject Recurrence)

为压力测试身份一致性,每个叙事让特定主体在**不同间隔**复现,且跨越多样上下文(剧情、背景、角色动作、相机视角)。这种变化确保高一致性分数反映**鲁棒身份学习**而非简单像素记忆。

Figure 22 展示了相关/无关历史的构造:相关历史是同一主体(老钟表匠)的不同 shot(Shot 1,6,11,16),无关历史是语义无关的干扰图(工具、齿轮、茶壶、猫)。

![Figure 22: 相关 vs 无关视觉历史](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig22_relevant.png)

*Figure 22:上排为相关历史(同一钟表匠),下排为无关干扰图,右侧为生成目标(同一主体)。当无关历史存在时,注意力可能错配到视觉显著但语义无关的 token,导致身份退化或伪影。*

### 4.4 评测协议(Benchmark Protocol,§B.2)

多轴评测,judge 模型为 **GPT-5.2**:

**逐图质量与 prompt 遵循:**
| 指标 | 类型 | 说明 |
|------|------|------|
| **HPS v3** | 学习的人类偏好模型 | 在大规模偏好标注上训练,输出与人类整体质量判断相关的标量,越高越好 |
| **PickScore** | CLIP-based 偏好模型 | 在 Pick-a-Pic 真实用户成对偏好上训练,作为 prompt 遵循 + 感知质量代理 |
| **LMM judge(视觉质量)** | GPT-5.2 reference-free | 评(a)视觉保真/伪影,(b)细节丰富度与构图,(c)文本描述遵循;固定 rubric,确定性解码(temp=0),多 prompt/seed 聚合 |

**长程一致性:**
| 指标 | 类型 | 说明 |
|------|------|------|
| **Subject identity(DINOv2)** | 自监督 ViT embedding | 计算主体 crop 与参考锚图的 cosine 相似度,越高身份保持越好 |
| **Subject identity(LMM)** | GPT-5.2 | reference-based 比较,聚焦面部/形态/标志特征,忽略 pose/视角/光照/背景变化 |
| **Style consistency** | GPT-5.2 | 评跨序列风格一致(色板、渲染/笔触、阴影、整体色调),标准化 rubric |

**评测 prompt 模板(论文 Figure 26-28 逐字给出):**

- **Prompt C.1(Subject Consistency):** "You are an expert at evaluating character identity consistency... Focus ONLY on Character Identity: Facial features, Hair, Accessories, Body type, Skin tone. MUST IGNORE: Lighting, Color grading, Camera angle, Shot type, Pose/Expression, Background, Clothing, Image quality. Score 1.0-10.0... Output ONLY valid JSON: {\"analysis\": \"...under 40 words\", \"score\": X.X}"
- **Prompt C.2(Style Consistency):** "You are an expert art critic... Focus ONLY on Artistic Style: Rendering style, Color palette, Brushwork/Texture, Shading, Line work, Overall aesthetic. MUST IGNORE: Subject matter, Lighting, Color temperature, Camera angle, Characters/Objects, Scene... Score 1.0-10.0... Output JSON: {\"analysis\": \"xxx (60-80 words)\", \"score\": X.X}"
- **Prompt C.3(Visual Quality):** "...evaluate the visual quality... Sharpness, Detail richness, Color harmony, Composition, Artifact detection, Overall aesthetic... Score 1.0-10.0... Output JSON: {\"analysis\": \"...under 50 words\", \"score\": X.X}"

### 4.5 已知偏置/局限

作者在 §A.5 承认:基准依赖"一次性注意力探针准确反映生成需求"的假设;对极度模糊的 prompt 或弱初始 grounding,相关性信号可能噪声大。$K_{\text{grd}}, K_{\text{img}}$ 是超参,虽稳定但最优值可能因模型架构而异。

---

## 5. 实验与评测

### 5.1 实验设置(§6.1)

- **基座:** BAGEL(hybrid AR+diffusion,有 ViT 理解分支 + VAE 生成分支)。
- **基准:** 50 个故事模板,每个 40 图槽位,fixed-text 协议。
- **CFG 配置(§C.2):** 双分支 CFG,三个 KV-cache 上下文——full($\mathcal{C}$)、text-CFG($\mathcal{C}_{\text{text}}$,缺最近生成文本段)、image-CFG($\mathcal{C}_{\text{img}}$,排除所有图像条件 token)。两个 guidance scale:`cfg_text_scale`=4.0,`cfg_img_scale`=1.5;`cfg_interval`=[0.4,1.0];`num_timesteps`=50;`timestep_shift`=3.0。
- **泛化测试:** 还在纯 AR 模型 **Lumina-mGPT**(基于 Chameleon)上验证(§D)。

### 5.2 主结果与设计空间消融(Table 1,§6.2)

Table 1 是全文核心实验,结构镜像表格分块(A 基线 / B 探针设计 / C 保留什么 / D 丢弃 vs 压缩 / E 最终配置)。

![Table 1: 长程交错生成的系统消融](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_table1_main.png)

**完整复现 Table 1(粗体为 UniLongGen 最终配置):**

| Variant | Signal | Unit | Budget | Discard | HPS v3↑ | PickScore↑ | Qual.↑ | ID↑ | Style↑ | DINOv2↑ |
|---------|--------|------|--------|---------|---------|-----------|--------|-----|--------|---------|
| **A. 基线** |||||||||||
| Base: Dense KV | – | – | – | – | 3.1677 | 0.1943 | 5.83 | 5.49 | 3.99 | 0.3164 |
| Sliding window (first+last N) | – | Image | N=4 | Drop | 4.4886 | 0.1969 | 6.47 | 5.73 | 4.56 | 0.3337 |
| Sliding window | – | Image | N=8 | Drop | 3.9126 | 0.1957 | 6.31 | 5.41 | 4.50 | 0.3267 |
| Sliding window | – | Image | N=12 | Drop | 4.0844 | 0.1960 | 6.34 | 5.82 | 4.76 | 0.3235 |
| **B. 探针设计**(固定 Unit=Image, Budget=K=4, Discard) |||||||||||
| B1: Text | Text | Image | K=4 | Drop | 7.5401 | 0.2031 | 7.56 | 7.01 | 5.64 | 0.4075 |
| B2: ViT | ViT | Image | K=4 | Drop | 5.5275 | 0.1989 | 6.90 | 5.96 | 4.85 | 0.3451 |
| B3: VAE | VAE | Image | K=4 | Drop | 3.1640 | 0.1940 | 6.00 | 4.48 | 3.61 | 0.2718 |
| **B4: layer-split: Text→VAE** | Hybrid | Image | K=4 | Drop | 7.5701 | 0.2025 | 7.58 | 7.09 | 6.13 | 0.4272 |
| **C. 保留什么**(固定 Signal=Hybrid) |||||||||||
| C1: Image-level retention | Hybrid | Image | K=4 | Drop | 7.5701 | 0.2025 | 7.58 | 7.09 | 6.13 | 0.4272 |
| C2: Image-level | Hybrid | Image | K=8 | Drop | 7.3491 | 0.2022 | 7.45 | 6.95 | 5.97 | 0.4204 |
| C3: Image-level | Hybrid | Image | K=12 | Drop | 6.9994 | 0.2013 | 7.36 | 6.72 | 6.04 | 0.3853 |
| C4: Token-level retention | Hybrid | Token | K=4 | Drop | 4.6454 | 0.1974 | 6.67 | 5.68 | 4.45 | 0.3598 |
| C5: Token-level | Hybrid | Token | K=8 | Drop | 4.3463 | 0.1967 | 6.65 | 5.70 | 4.52 | 0.3431 |
| C6: Token-level | Hybrid | Token | K=12 | Drop | 4.4805 | 0.1967 | 6.61 | 5.86 | 4.67 | 0.3537 |
| C7: Grouped token | Hybrid | Tok×8 | K=4 | Drop | 4.5345 | 0.1970 | 6.60 | 5.79 | 4.25 | 0.3619 |
| C8: Grouped token | Hybrid | Tok×32 | K=4 | Drop | 4.6553 | 0.1971 | 6.63 | 5.69 | 4.38 | 0.3558 |
| C9: Grouped token | Hybrid | Tok×128 | K=4 | Drop | 4.6371 | 0.1974 | 6.63 | 5.67 | 4.33 | 0.3512 |
| **D. 丢弃 vs 压缩**(固定 Signal/Unit=Image/Budget=K=4) |||||||||||
| D1: Drop discarded tokens | Hybrid | Image | K=4 | Drop | 7.5701 | 0.2025 | 7.58 | 7.09 | 6.13 | 0.4272 |
| D2: Downsample (AvgPool) | Hybrid | Image | K=4 | Comp ×4 AvgPool | 6.9601 | 0.2018 | 7.36 | 6.92 | 5.32 | 0.3997 |
| D3: Downsample (MaxPool) | Hybrid | Image | K=4 | Comp ×4 MaxPool | -0.527 | 0.1895 | 4.71 | 3.85 | 3.24 | 0.2078 |
| D5: Downsample (interp) | Hybrid | Image | K=4 | Comp ×4 LERP | 7.1506 | 0.2019 | 7.47 | 7.01 | 5.79 | 0.3973 |
| D6: Downsample (interp) | Hybrid | Image | K=4 | Comp ×8 LERP | 7.3643 | 0.2022 | 7.54 | 6.94 | 5.55 | 0.4124 |
| D7: Downsample (interp) | Hybrid | Image | K=4 | Comp ×16 LERP | 7.4855 | 0.2025 | 7.57 | 7.03 | 5.88 | 0.4118 |
| **E. 最终配置** |||||||||||
| **★ Ours (UniLongGen)** | **Text@L1+VAE@Lₛyn** | **Image** | **K=4** | **Drop** | **7.5701** | **0.2025** | **7.58** | **7.09** | **6.13** | **0.4272** |

**逐块解读:**

**(A)为什么 dense KV 失败、recency window 脆弱。** Dense KV 在长程崩溃(HPS v3=3.17,DINOv2=0.316),印证"更多历史可能主动有害"。滑窗(只保最近 N 图)部分缓解(减少可见图像事件),但**脆弱**——它与长程剧情约束错配,容易丢掉早期身份/风格锚点或非最近但高价值的参考。

**(B)一个探针不够,深度很重要。** Text@Layer1 大幅提升(B1:HPS v3=7.54),说明稳定 grounding 能减少欠条件化。但信号必须匹配深度专业化:ViT@Layer1 较弱(B2),VAE@Layer1 失败(B3,因浅层 VAE 交互被高方差 patch 统计/尾部离群主导)。**Text→VAE 层分割(B4)** 最佳平衡质量与一致性(HPS v3=7.57,Style=6.13),显式遵循 grounding→synthesis 的层间转换。

**(C)保留什么:事件级选择匹配瓶颈、消除尾部风险。** Image-level 保留(C1-C3)持续优于 token/grouped-token 保留(C4-C9),直接支持 Event Bottleneck——稳定性由保留多少**图像事件**决定,而非保留任意 token 子集。Token-level 剪枝留下密集视觉 key 碎片(和结构路由 token),仍产生离群值、破坏 Softmax 分配。还观察到**小预算最优**:$K$ 超过 4 反而重新引入竞争者、略微降质。

**(D)丢弃胜过压缩。** 给定相同选中参考,**直接丢弃最可靠**。压缩(池化/插值)可能保留虚假竞争者并引入分布偏移;MaxPool 灾难性结果(D3:HPS v3=-0.527)与"放大高频伪影→劫持注意力"的核心机制一致。整体支持核心论点:**删除竞争视觉事件比总结它们更安全**。

### 5.3 VAE 探针层消融(Table 2,§6.2-E)

固定 Hybrid Text→VAE、$K=4$、Drop,变化 $\ell_{\text{syn}} \in \{10,15,20\}$:

| Variant | HPS v3↑ | PickScore↑ | Qual.↑ | Style Cons.↑ | DINOv2↑ |
|---------|---------|-----------|--------|--------------|---------|
| Dense KV (Bagel) | 3.1677 | 0.1943 | 5.83 | 3.99 | 0.3164 |
| Sliding window | 4.4886 | 0.1969 | 6.47 | – | 0.3337 |
| Ours, $\ell_{\text{syn}}$=10 | 7.4349 | 0.2019 | 7.53 | 6.12 | 0.4251 |
| **Ours, $\ell_{\text{syn}}$=15** | **7.5701** | **0.2025** | **7.58** | **6.13** | **0.4272** |
| Ours, $\ell_{\text{syn}}$=20 | 7.4920 | 0.2023 | 7.52 | 6.03 | 0.4239 |

性能在后期层鲁棒,$\ell_{\text{syn}}=15$ 附近最优——太早则相关性信号噪声大(tail-risk patch 匹配),太晚则模型可能过拟合步特定细节。

### 5.4 语义 oracle vs 注意力选择(Table 3,§6.6)

一个直觉基线是用外部(人类/oracle)标准保留"最相关"历史图。**令人意外的是,这种语义保留反而不如基于注意力的选择:**

| Variant | HPS v3↑ | Qual.↑ | Style↑ | DINOv2-ID↑ |
|---------|---------|--------|--------|-----------|
| Dense KV (no pruning) | 3.1677 | 5.83 | 3.99 | 0.3164 |
| Semantic oracle (top-K by human relevance) | 5.1158 | 6.77 | 5.1 | 0.4205 |
| Text-block matching (current→history, top-K) | 1.7063 | 5.35 | 3.06 | 0.3094 |
| **UniLongGen (attention-based top-K)** | **7.5701** | **7.58** | **6.13** | **0.4272** |

**为什么 oracle 相关性会有害:**(1)相关性不纯语义——生成可能依赖近期上下文的 pose/布局/光照连续性,或最匹配当前轨迹的特定参考视角;(2)硬 mask 的分布偏移——任何剪枝都改变 Softmax 归一化、重路由注意力,强制人定义子集会放大虚假依赖。

> **Implication 3:模型对齐胜过语义直觉。** 模型对齐选择保留模型自身偏好的历史图像块子集,产生更好的长程保真。这揭示了**语义相关性与生成效用之间的鸿沟**。

### 5.5 效率对比(Figure 12,§6.4)

![Figure 12: 运行时随历史上下文的 scaling](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig12_runtime.png)

报告平均可见 KV 长度(KV%)和端到端单图墙钟时间(关闭 CFG)。Dense KV 时间随历史 token 近似线性增长(每步都 attend 整个历史);UniLongGen 驱逐非选中历史、只保留小而固定的可见集,运行时基本对原始历史长度不敏感:

- **512×512,100k token:** Dense KV 97.2s vs UniLongGen 8.5s
- **1024×1024,350k token:** Dense KV 363s vs UniLongGen 31.6s — **最高约 11× 加速**

### 5.6 定性结果(Figure 11,§6.3)

![Figure 11: 长程稳定性的定性对比](https://raw.githubusercontent.com/bugWholesaler/paper_reading/main/figures/unilonggen_fig11_qualitative.png)

*Figure 11:同一 40 图序列在后期位置(27/29/32/36/39)、768×432 分辨率的快照对比。行依次为 Bagel(dense KV)、Ours、Sliding Window(4/12 块)、Token-Level(1/128 token/组)、Semantic Oracle、Text-To-Text Retrieval。Dense KV 和多个启发式在后期退化为伪影或漂移,UniLongGen 保持连贯合成。*

论文还提供 7 组完整 40 图定性结果(Figure 13-19),涵盖电影分镜、Pixar 风动画、机器人、沙漠、灯塔等多种叙事,展示长程身份/风格一致性。

### 5.7 泛化到纯 AR 模型(Table 4,§D)

在 **Lumina-mGPT**(Chameleon-based,纯 AR)上验证:

| Model | Method | HPS v3↑ | Qual.↑ |
|-------|--------|---------|--------|
| Lumina-mGPT | Dense KV | 2.7954 | 0.1904 |
| Lumina-mGPT | Sliding Window | 7.2965 | 0.2014 |
| Lumina-mGPT | + UniLongGen | 7.3020 | 0.2015 |

**增益微弱但有解释力**(作者诚实讨论):(1)**失败模式因生成范式而异**——hybrid AR+diffusion 每张图经多步精炼,密集视觉历史反复进入条件化通路,竞争持续;纯 AR 逐 token 解码,更被当前上下文/自注意力主导(Figure 23),recency 截断已能移除多数竞争者,滑窗就够强。(2)**增益取决于模型对交错生成的容量**——Lumina-mGPT 多图能力本就弱,短序列都几乎无跨图一致性,长程退化由内在能力差距而非注意力分配瓶颈主导。(3)给出**跨模型迁移配方**:UniLongGen 机制不绑定固定层分割或固定 query token,而是**对齐模型自身注意力签名**——实践中可通过测量层级模态比来诊断签名,选择对应探针层和 query 锚点。注意:Lumina-mGPT 的注意力签名与 BAGEL **相反**(早期层强调图像,深层转向文本),所以探针用 image-start special token query + Layer 1 图像选择。

### 5.8 Pre-Softmax vs Post-Softmax(Table 5,§F.2)

| Variant | HPS v3↑ | PickScore↑ | Style Cons.↑ | DINOv2-ID↑ |
|---------|---------|-----------|--------------|-----------|
| Baseline | 3.1677 | 0.1943 | 3.99 | 0.3164 |
| Post-Softmax | 5.2779 | 0.1978 | 5.44 | 0.3617 |
| **Pre-Softmax (Ours)** | **7.5701** | **0.2025** | **6.13** | **0.4272** |

**为什么 Post-Softmax 偏向 recency:** Softmax 后注意力 top-heavy、被少数最高分 key 主导;长历史里最近 turn 的 token 最易匹配(受中间上下文降解最少),常获最大注意力。于是 Post-Softmax 相关性估计塌缩成近-recency 启发式。Pre-Softmax QK 分数显著改善长程质量和身份一致性,缓解 recency 偏置。

### 5.9 统计可靠性与失败案例

- **统计:** 评测在多个随机种子上平均(§3.1),但论文未给置信区间/标准差的明确数值(Figure 2 用 shaded band 表示方差)。
- **失败案例:** §A.5 承认极模糊 prompt 或弱初始 grounding 下相关性信号可能噪声大;Figure 11 的 Text-To-Text Retrieval 行展示纯文本匹配基线的崩溃(HPS v3=1.71)。

---

## 6. 优点(Strengths)

1. **诊断深度罕见且自洽。** 论文不止给方法,而是建立了完整的机制链条:Event Bottleneck(Figure 3)→ 模态不对称(Figure 4)→ 重尾劫持 + Softmax 放大(Figure 6-7)→ 深度专业化(Figure 8-9)。每个诊断都直接催生一个设计决策,形成"现象→机制→对策"的闭环。

2. **训练-free 且即插即用,效率收益实打实。** 无需任何微调即可把 HPS v3 翻倍以上(3.17→7.57),同时实现最高 11× 加速(Figure 12)——大多数上下文管理方法是质量/效率二选一,这里两者兼得,因为驱逐既去污染又减计算。

3. **消融极其系统,反直觉发现有说服力。** Table 1 沿单轴变化做受控对比,覆盖探针信号/单元/预算/丢弃方式四个维度;Table 3 的"语义 oracle 反而更差"和 Table 5 的"Pre-Softmax 胜 Post-Softmax"都是有价值的反直觉结论,支撑"模型对齐 > 语义直觉"的核心主张。

---

## 7. 弱点与局限(Weaknesses & Limitations)

1. **泛化性证据偏弱。** 核心增益高度依赖 BAGEL 的 hybrid AR+diffusion 特性和 ViT/VAE 双表示;在纯 AR 的 Lumina-mGPT 上增益几乎可忽略(Table 4,7.2965→7.3020)。作者虽诚实解释,但这意味着方法的价值面相对窄——只有当模型(i)合成期密集视觉历史持续竞争注意力,且(ii)本就有能力利用历史参考时才显著有效。

2. **超参敏感性与可迁移成本。** $\ell_{\text{syn}}$、$\ell_{\text{grd}}$、$K_{\text{img}}$、$K_{\text{grd}}$ 都是需调的超参,且作者承认最优值因架构而异;跨模型迁移需要先诊断注意力签名再选探针层,缺乏自动化方案,工程落地有摩擦。

3. **评测可靠性披露不足。** 主表(Table 1-5)均为单点数值,未报告标准差/置信区间;质量与一致性的核心指标(Qual./ID/Style)依赖 GPT-5.2 作为 judge,存在 LLM-judge 自身偏置和不可复现风险(GPT-5.2 是闭源模型)。50 个模板的基准规模也偏小。

---

## 8. 与并行工作对比

| 工作 | 问题框定 | 模型/规模 | 核心机制 | 是否训练 | 代码/权重 |
|------|----------|-----------|----------|----------|-----------|
| **UniLongGen(本文)** | 长程交错生成的注意力劫持 | BAGEL(hybrid) | 推理期 KV 驱逐 + 双探针 | ❌ 训练-free | ❌ |
| [VINO (Chen et al., 2026)](https://arxiv.org/abs/2601.02358) | 交错全模态上下文 | 自研统一模型 | 架构层面扩展上下文 | ✅ 需训练 | 见原文 |
| [NextFlow (Zhang et al., 2026)](https://arxiv.org/abs/2601.02204) | 统一序列建模激活理解+生成 | 自研 | 统一序列建模 | ✅ | 见原文 |
| 通用 KV 压缩(H2O/StreamingLLM 类) | 长上下文显存 | LLM | token 压缩/稀疏 | ❌ | 多数开源 |

UniLongGen 与通用 KV 压缩的本质区别:后者目标是"在预算内塞进更多信息",前者主张"无关密集视觉历史不仅浪费、而是主动有害",因此选择 relevance-filtered sparsity 而非 total recall。

---

## 9. 复现性审计(Reproducibility Audit)

| 项 | 是否释出 | 备注 |
|----|----------|------|
| 代码 | ❌ | 论文未给 GitHub 链接 |
| 权重 | ❌ | 训练-free,但基座 BAGEL 权重需自行获取 |
| 训练数据 | N/A | 训练-free,无训练数据 |
| 评测数据 | ❌ | 50 个叙事模板未明确开源 |
| 超参 | ✅ | $\ell_{\text{grd}}=1, \ell_{\text{syn}}=15, K\approx4$,CFG 参数(§C.2)完整给出 |
| 评测/judge prompt | ✅ | Figure 26-28 逐字给出三个 GPT-5.2 prompt |
| 硬件规格 | ❌ | 未明确 GPU 型号/数量;只报告相对墙钟时间 |

**复现性结论:** 这是一篇**方法可复现性中等、产物可复现性偏低**的论文。方法本身描述清晰——核心公式(1-3)、双探针层、KV 驱逐策略、CFG 配置和全部 judge prompt 都给足了,一个熟悉 BAGEL 的工程师可以据此重写。但**代码、评测基准、权重、硬件均未释出**,且核心质量指标依赖闭源 GPT-5.2,使得**精确数值复现困难**。考虑到这是 2026 年 3 月的 preprint,后续可能补充开源。总体:理解和重实现门槛低,但逐点数值对齐门槛高。

---

## 总结

UniLongGen 的真正贡献不在那个三步流程本身,而在**重新定义了问题**:长程交错生成崩溃不是"长上下文"问题,而是 **Softmax 注意力预算下的事件级零和竞争**——密集视觉历史会通过重尾离群值**主动劫持**注意力、污染合成。一旦接受这个框架,解法就顺理成章:**删除竞争源(丢弃而非压缩)、按层功能分工选择(文本 grounding 用早期层、视觉合成用后期层)、信任模型自身的注意力信号(模型对齐胜过语义 oracle)**。这三条原则——加上 Pre-Softmax 打分对抗 recency 偏置——构成了一个干净、可解释、训练-free 且高效(11× 加速)的方案。它的主要软肋是泛化性:只有在"密集视觉历史持续竞争 + 模型本就能用历史参考"的 hybrid 模型上才显著有效,在纯 AR 模型上几乎无增益。
