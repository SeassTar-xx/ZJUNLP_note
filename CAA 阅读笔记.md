# CAA 阅读笔记

## 1. 基本信息
- 论文标题：Steering Llama 2 via Contrastive Activation Addition
- 方法名称：Contrastive Activation Addition (CAA)
- 方向关键词：Activation Steering, Steering Vector, Residual Stream, Contrastive Examples, Mean Difference, System Prompting, Finetuning, Interpretability
- 阅读日期：2026-06-06

## 2. 这篇论文想解决什么问题？
- 核心目标：
  - 提出 CAA，用 activation-level intervention 来 steer language model behavior。
  - 与直接改 weights 的 finetuning 不同，CAA 在 forward pass 中修改 residual stream activations。
- 基本思路：
  - 用 positive / negative contrastive examples 构造某种 behavior 的 steering vector。
  - inference time 时，把这个 steering vector 加到 user prompt 之后所有 token positions 的 residual stream 上。
  - multiplier 可以为正或负，用来增强或削弱目标 behavior。
- 总结：
  - CAA 可以看作一种轻量的 activation steering 方法：它利用 contrastive examples 提取 behavior direction，然后在 inference time 通过 residual stream addition 改变模型输出倾向。

## 3. Figure 1：CAA 的生成与应用
![Figure 1: CAA generation and application](assets/caa/caa-figure1.png)

- Figure 1(a)：steering vector generation
  - 构造 contrast pairs：一个 positive example of behavior，一个 negative example of behavior。
  - 对每个 pair 做 forward pass，在指定 layer `L`、answer letter 的 token position 提取 residual stream activations。
  - 对 positive activation 减 negative activation，再对 many contrast pairs 求平均。
- Figure 1(b)：steering vector application
  - inference time 时，把乘上 multiplier 的 steering vector 加回 residual stream。
  - 添加位置是 instruction / user prompt 之后的所有 token positions。
  - 正 multiplier 通常让输出更偏向 positive behavior，负 multiplier 通常让输出更偏向 negative behavior。

## 4. Method：CAA 怎么计算 steering vector？
- 给定 dataset `D`，其中每个样本是 `(prompt p, positive completion c_p, negative completion c_n)`。
- 在 layer `L` 上计算 Mean Difference vector：
  - `v_MD = (1 / |D|) * sum(a_L(p, c_p) - a_L(p, c_n))`
  - `a_L(...)` 表示 layer `L` 的 activation。
- 我的理解：
  - 这个公式就是把 positive samples 在第 `L` 层的 residual stream activation 减去 negative samples 在同一层的 residual stream activation。
  - 因为 paired prompts 只改变 answer option，prompt 其他部分保持不变，所以差分更可能隔离出与 target behavior 相关的方向。
- layer sweep：
  - 在第 `L` 层提取 steering vector。
  - 在 held-out multiple-choice questions 上测试 steering 效果。
  - 用 multiplier `+1` 和 `-1` 看是否分别增强 / 削弱 target behavior。
  - 找 steering effect 最明显的 layer。

## 5. Figure 3 / Figure 4：在哪一层加 vector 最有效？
![Figure 3: Layer sweeps](assets/caa/caa-figure3.png)

- Figure 3 展示 layer sweep 结果：
  - 目标是寻找在哪一层 residual stream 上加 steering vector 最有效。
  - 在 Llama 2 7B Chat 中，效果大致集中在 layer 13 附近。
  - 在 Llama 2 13B Chat 中，最佳层通常在 layer 14 或 15 附近。
- 我的理解：
  - 不同 behavior 的最佳层不完全相同，但整体上存在一个中间层附近的 effective range。
  - 这说明 behavior representation 可能不是在最早层或最后层才出现，而是在中间层逐渐形成并可被 steering。

![Figure 4: Multiplier effects](assets/caa/caa-figure4.png)

- Figure 4 在最佳层附近探索不同 multiplier 的效果。
- 大多数 behavior 随 multiplier 增大而增强，说明 CAA 的 steering strength 在一定范围内可以连续调节。
- 但这不意味着 multiplier 越大越好；论文后面提到过大的 multiplier 会影响 open-ended generation quality。

## 6. CAA and system-prompting
- 论文比较了 CAA 和 system-prompting：
  - system prompt 通过外部 instruction 引导模型。
  - CAA 通过 internal activation-level control 引导模型。
- 重点：
  - CAA 可以在 system prompt 的基础上继续改变模型行为。
  - 因此 CAA 和 prompt engineering 不是完全重复的。
  - CAA 可能提供额外的 internal control，尤其是 multiplier 带来的更连续的 steering quantity。
- 我的理解：
  - 这只能说明在作者实验设置中，CAA 能补充 system-prompting。
  - 它不等价于 CAA 在所有任务上都比 prompt engineering 更好。

## 7. Comparison to finetuning
- Section 6 比较 CAA 与 supervised finetuning。
- 实验设置：
  - 作者在与 CAA 相同的 multiple-choice behavioral datasets 上对 Llama 2 7B Chat 做 positive 或 negative supervised finetuning。
  - finetuning 目标是让模型更倾向于选择 target behavior 或 opposite behavior 对应的 answer token。
- 主要观察：
  - supervised finetuning 在 held-out multiple-choice setting 中非常有效。
  - finetuning 对 open-ended generation 也有一定影响。
  - 在 7 个 behavior 中，有 3 个上 CAA 可以在 finetuning 之后继续增强或削弱 target behavior。
- 需要谨慎的地方：
  - CAA 与 finetuning 的组合不是简单线性叠加。
  - 例如 Refusal 中，positive steering 加在 finetuned model 上反而降低 refusal score。
  - Sycophancy 上出现了从 multiple-choice 到 open-ended generation 的 OOD generalization failure；CAA 在这些设置中仍能泛化，可能说明 CAA steering existing learned representations 时有一定开放式迁移优势。
- 成本对比：
  - finetuning 需要 backward pass 和更多 GPU 资源。
  - CAA vector generation 只需要 forward pass，计算成本更低。
- 总结：
  - Section 6 的结论不是 CAA 全面优于 finetuning。
  - 更准确地说，CAA 更轻量、可叠加，并且在某些 open-ended generation / OOD setting 中可能比基础 supervised finetuning 更有泛化优势。

## 8. Understanding and interpreting CAA

### 8.1 Similarity between steering vectors and per-token activations
![Figure 6: Per-token activation similarity](assets/caa/caa-figure6.png)

- 论文把 steering vector 和 regular per-token activations 做 cosine similarity。
- 目的：
  - 看 steering vector 是否不仅能 steer，还能作为 behavior feature probe。
  - 如果某些 token 的 activation 和 steering vector 相似，可能表示这些 token 中包含相应 behavior tendency。
- Figure 6 的理解：
  - refusal vector 不只是“随便一个向量”，它能在 token level 上检测出 refusal behavior 是否在当前 token 中出现。
  - steering vector 和 per-token activations 的 similarity，能够反映 target behavior 在具体 token 中的“存在程度”。
- 我的理解：
  - 这提供了一定 interpretability evidence。
  - 但 cosine similarity 只是一个 probe，不应直接等同于完整 causal mechanism。

### 8.2 Similarity between vectors generated at different layers
![Figure 7: Inter-layer cosine similarity](assets/caa/caa-figure7.png)

- Figure 7 展示不同 layer 生成的 steering vectors 之间的 cosine similarity。
- 颜色越红，表示两个 layer 提取出的 steering vector 越相似。
- 论文观察到：
  - 相邻层 vector 更相似。
  - 后半段层之间的 similarity decline 更慢。
- 我的理解：
  - 当模型抽取出某种 high-level abstract behavior representation 后，这个表示可能在后续层中逐渐稳定。
  - 因此 behavior direction 可能在中间层逐渐出现，并在后续层保持某种稳定性。

![Figure 8: Transfer layer 13 vector](assets/caa/caa-figure8.png)

- Figure 8 测试从 layer 13 提取的 vector 加到其他 layer 是否仍然有效。
- 观察：
  - layer 13 vector 加到别的层也有一定 steering effect。
  - 这说明 CAA direction 不是完全 layer-specific 的，它可能代表某种较 general 的 behavior representation。
  - 但效果不能无限跨层使用；后面某些层的 representation space 或 processing stage 可能已经变化。

![Figure 9: Base vs chat model vector similarity](assets/caa/caa-figure9.png)

- Figure 9 比较 base model 和 chat model 的 steering vectors。
- 观察：
  - early layers 和 late layers 的相似度差异较大。
  - 中间层存在一个 similarity peak。
- 我的理解：
  - 这可能说明 RLHF / chat tuning 改变了部分层的 behavior representation。
  - 但在某些中间层，base model 和 chat model 的 representation 仍可能存在较高相似性。