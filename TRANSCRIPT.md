# Transcript

Both exchanges, verbatim. Each block is the exact text sent to or returned by `openai/gpt-4`.

## 1. Predicting the result

### User prompt

~~~~text
I've been developing a pipeline that layers mathematical instruction tuning and then reinforcement learning on top of a 7B DeepSeekMath-Base model, and I want to lay out exactly how it's built before digging further.

For the instruction-tuning stage, I construct a mathematical instruction-tuning dataset covering English and Chinese problems from different mathematical fields and varying complexity levels, pairing problems with solutions in chain-of-thought (CoT), program-of-thought (PoT), and tool-integrated reasoning format. The total number of training examples is 776K. This gives me DeepSeekMath-Instruct 7B, which undergoes mathematical instruction tuning based on DeepSeekMath-Base. Training examples are randomly concatenated until reaching a maximum context length of 4K tokens, and I train the model for 500 steps with a batch size of 256 and a constant learning rate of 5e-5.

For the RL stage I use a variant of Proximal Policy Optimization (PPO). PPO is an actor-critic RL algorithm that is widely used in the RL fine-tuning stage of LLMs. In particular, it optimizes LLMs by maximizing the following surrogate objective:
$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right)  A_{t} \right],$$
where $\pi_{\theta}$ and $\pi_{\theta_{old}}$ are the current and old policy models, and $q, o$ are questions and outputs sampled from the question dataset and the old policy $\pi_{\theta_{old}}$, respectively. $\epsilon$ is a clipping-related hyper-parameter introduced in PPO for stabilizing training. $A_t$ is the advantage, which is computed by applying Generalized Advantage Estimation (GAE), based on the rewards $\{r_{\ge t}\}$ and a learned value function $V_{\psi}$. Thus, in PPO, a value function needs to be trained alongside the policy model, and to mitigate over-optimization of the reward model, the standard approach is to add a per-token KL penalty from a reference model in the reward at each token, i.e., $r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})}$, where $r_\phi$ is the reward model, $\pi_{ref}$ is the reference model, which is usually the initial SFT model, and $\beta$ is the coefficient of the KL penalty.

As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction, while in the LLM context usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token. To address this, my variant obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. More specifically, for each question $q$, it samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$ from the old policy $\pi_{\theta_{old}}$ and then optimizes the policy model by maximizing the following objective:
$$\mathcal{J}(\theta) = \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\},$$
where $\epsilon$ and $\beta$ are hyper-parameters, and $\hat{A}_{i,t}$ is the advantage calculated based on relative rewards of the outputs inside each group only. This group-relative way of calculating the advantages aligns with the comparative nature of reward models, as reward models are typically trained on datasets of comparisons between outputs on the same question. Also note that, instead of adding the KL penalty in the reward, I regularize by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $\hat{A}_{i,t}$. And different from the per-token KL penalty used in PPO above, I estimate the KL divergence with the following unbiased estimator:
$$\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1,$$
which is guaranteed to be positive.

The advantages can be computed with outcome or process supervision. With outcome supervision, a reward model scores the $G$ outputs, yielding rewards $\mathbf{r}=\{r_1, r_2, \cdots, r_G\}$; these are normalized by subtracting the group average and dividing by the group standard deviation, and the advantages of all tokens in output $o_i$ are set to the normalized reward, i.e., $\hat{A}_{i, t} = \widetilde{r}_i = \frac{r_i- {\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}$. With process supervision, a process reward model scores each reasoning step, yielding $\mathbf{R} = \{ \{r_1^{index(1)},\cdots,r_1^{index(K_1)}\}, \cdots, \{r_G^{index(1)},\cdots,r_G^{index(K_G)}\} \}$, where $index(j)$ is the end token index of the $j$-th step and $K_i$ is the total number of steps in the $i$-th output; these are normalized as $\widetilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - {\rm mean(\mathbf{R})}}{{\rm std(\mathbf{R})}}$, and the advantage of each token is the sum of the normalized rewards from the following steps, $\hat{A}_{i, t} = \sum_{index(j) \ge t} \widetilde{r}_i^{index(j)}$.

On top of this instruction-tuned model, I optionally conduct RL based on DeepSeekMath-Instruct 7B. The training data for RL are chain-of-thought-format questions related to GSM8K and MATH from the SFT data, consisting of around 144K questions. I exclude the other SFT questions so I can investigate the impact of RL on benchmarks that lack data throughout the RL phase. I construct the training set of reward models following the approach in prior work, and train my initial reward model based on DeepSeekMath-Base 7B with a learning rate of 2e-5. For this RL algorithm, I set the learning rate of the policy model as 1e-6, with a KL coefficient of 0.04. For each question, I sample 64 outputs. The max length is set to 1024, and the training batch size is 1024. The policy model only has a single update following each exploration stage. I evaluate the resulting DeepSeekMath-RL 7B model on the same benchmarks as DeepSeekMath-Instruct 7B; for this RL model, GSM8K and MATH with chain-of-thought reasoning can be regarded as in-domain tasks, while all the other benchmarks can be regarded as out-of-domain tasks.

I'm evaluating all of this under both chain-of-thought-only and tool-integrated reasoning inference settings. My baseline, from DeepSeekMath-Instruct 7B before any RL, gives me the following Top1 accuracy numbers. Under chain-of-thought inference: GSM8K = 82.9%, MATH = 46.8%, MGSM-zh = 73.2%, CMATH = 84.6%. Under tool-integrated inference: GSM8K = 83.7%, MATH = 57.4%, MGSM-zh = 72.0%, CMATH = 84.3%.

My question about this setup: after switching from DeepSeekMath-Instruct 7B (my baseline above) to DeepSeekMath-RL 7B, predict the resulting value for each: GSM8K (Chain-of-Thought) Top1 accuracy (in %); MATH (Chain-of-Thought) Top1 accuracy (in %); MGSM-zh (Chain-of-Thought) Top1 accuracy (in %); CMATH (Chain-of-Thought) Top1 accuracy (in %); GSM8K (Tool-Integrated) Top1 accuracy (in %); MATH (Tool-Integrated) Top1 accuracy (in %); MGSM-zh (Tool-Integrated) Top1 accuracy (in %); CMATH (Tool-Integrated) Top1 accuracy (in %).
~~~~

### GPT-4 reply

~~~~text
Given the detailed setup of the experiment, it seems that the reinforcement learning (RL) stage is well-designed and could potentially improve the performance of the model. However, the impact of RL on large language models (LLMs) can be unpredictable and often depends on the specifics of the dataset and the RL algorithm used.

In this case, the RL stage is trained on a subset of the supervised fine-tuning (SFT) data, which might limit its ability to generalize to all tasks. However, the use of Proximal Policy Optimization (PPO), a widely used RL algorithm, and the specific modifications made to it, could potentially lead to improvements.

For the in-domain tasks (GSM8K and MATH), where the RL stage has direct training data, I would expect to see a noticeable improvement. For the out-of-domain tasks (MGSM-zh and CMATH), the improvements may be less pronounced, but still possible due to the generalization capabilities of the model.

For the tool-integrated reasoning inference, the impact of RL might be less clear, as it's not explicitly mentioned whether the RL stage was designed with tool-integrated reasoning in mind. However, given the general nature of RL and its ability to optimize for a given reward function, it's still possible that there could be improvements.

Here are my specific predictions for the Top1 accuracy metrics:

```json
{
"predictions": [
    {"slot": "GSM8K (Chain-of-Thought) Top1 accuracy (in %)", "value": 85.3},
    {"slot": "MATH (Chain-of-Thought) Top1 accuracy (in %)", "value": 49.2},
    {"slot": "MGSM-zh (Chain-of-Thought) Top1 accuracy (in %)", "value": 74.0},
    {"slot": "CMATH (Chain-of-Thought) Top1 accuracy (in %)", "value": 85.2},
    {"slot": "GSM8K (Tool-Integrated) Top1 accuracy (in %)", "value": 84.5},
    {"slot": "MATH (Tool-Integrated) Top1 accuracy (in %)", "value": 59.0},
    {"slot": "MGSM-zh (Tool-Integrated) Top1 accuracy (in %)", "value": 72.5},
    {"slot": "CMATH (Tool-Integrated) Top1 accuracy (in %)", "value": 84.8}
  ]
}
```
~~~~

## 2. Its take versus actor-critic PPO

### User prompt

~~~~text
I'm designing the reinforcement learning stage for fine-tuning a 7B language model on mathematical reasoning, after supervised fine-tuning. Here is the algorithm I'm considering.

I'm considering a variant of Proximal Policy Optimization (PPO). PPO is an actor-critic RL algorithm that is widely used in the RL fine-tuning stage of LLMs. In particular, it optimizes LLMs by maximizing the following surrogate objective:
$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right)  A_{t} \right],$$
where $\pi_{\theta}$ and $\pi_{\theta_{old}}$ are the current and old policy models, and $q, o$ are questions and outputs sampled from the question dataset and the old policy $\pi_{\theta_{old}}$, respectively. $\epsilon$ is a clipping-related hyper-parameter introduced in PPO for stabilizing training. $A_t$ is the advantage, which is computed by applying Generalized Advantage Estimation (GAE), based on the rewards $\{r_{\ge t}\}$ and a learned value function $V_{\psi}$. Thus, in PPO, a value function needs to be trained alongside the policy model, and to mitigate over-optimization of the reward model, the standard approach is to add a per-token KL penalty from a reference model in the reward at each token, i.e., $r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})}$, where $r_\phi$ is the reward model, $\pi_{ref}$ is the reference model, which is usually the initial SFT model, and $\beta$ is the coefficient of the KL penalty.

As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction, while in the LLM context usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token. To address this, my variant obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. More specifically, for each question $q$, it samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$ from the old policy $\pi_{\theta_{old}}$ and then optimizes the policy model by maximizing the following objective:
$$\mathcal{J}(\theta) = \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\},$$
where $\epsilon$ and $\beta$ are hyper-parameters, and $\hat{A}_{i,t}$ is the advantage calculated based on relative rewards of the outputs inside each group only. This group-relative way of calculating the advantages aligns with the comparative nature of reward models, as reward models are typically trained on datasets of comparisons between outputs on the same question. Also note that, instead of adding the KL penalty in the reward, I regularize by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $\hat{A}_{i,t}$. And different from the per-token KL penalty used in PPO above, I estimate the KL divergence with the following unbiased estimator:
$$\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1,$$
which is guaranteed to be positive.

The advantages can be computed with outcome or process supervision. With outcome supervision, a reward model scores the $G$ outputs, yielding rewards $\mathbf{r}=\{r_1, r_2, \cdots, r_G\}$; these are normalized by subtracting the group average and dividing by the group standard deviation, and the advantages of all tokens in output $o_i$ are set to the normalized reward, i.e., $\hat{A}_{i, t} = \widetilde{r}_i = \frac{r_i- {\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}$. With process supervision, a process reward model scores each reasoning step, yielding $\mathbf{R} = \{ \{r_1^{index(1)},\cdots,r_1^{index(K_1)}\}, \cdots, \{r_G^{index(1)},\cdots,r_G^{index(K_G)}\} \}$, where $index(j)$ is the end token index of the $j$-th step and $K_i$ is the total number of steps in the $i$-th output; these are normalized as $\widetilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - {\rm mean(\mathbf{R})}}{{\rm std(\mathbf{R})}}$, and the advantage of each token is the sum of the normalized rewards from the following steps, $\hat{A}_{i, t} = \sum_{index(j) \ge t} \widetilde{r}_i^{index(j)}$.

What is your take on this algorithm compared with traditional actor-critic methods that learn a value function, such as PPO with GAE? Please cover its strengths and weaknesses relative to them, any theoretical concerns (bias, variance, credit assignment, stability), what you expect to go wrong in practice, and whether you would expect it to outperform PPO, match it more cheaply, or underperform it. Be candid and specific.
~~~~

### GPT-4 reply

~~~~text
Your proposed algorithm is an interesting variant of Proximal Policy Optimization (PPO) that avoids the need for a learned value function by using group-relative rewards as a baseline. Below, I will analyze its strengths and weaknesses, compare it to traditional actor-critic methods like PPO with GAE, and discuss potential theoretical and practical concerns.

---

### **Strengths**

1. **Avoidance of Value Function Approximation**:
   - By using group-relative rewards as a baseline, your method avoids the need to train a separate value function. This eliminates the computational and memory overhead associated with maintaining a value function, which is particularly significant for large language models (LLMs).
   - This also sidesteps the challenges of value function instability, which can arise when the value function fails to accurately represent the expected returns, especially in the sparse-reward setting typical of LLM fine-tuning.

2. **Alignment with Reward Model Nature**:
   - Reward models for LLMs are often trained on pairwise comparisons of outputs for the same question. Your group-relative advantage calculation aligns well with this comparative nature, as it normalizes rewards within a group of sampled outputs. This could lead to better credit assignment and more stable training.

3. **Simplified Advantage Computation**:
   - By using normalized rewards as advantages, your method simplifies the computation of advantages compared to GAE, which requires estimating multi-step returns and bootstrapping with a value function.

4. **Direct KL Regularization**:
   - Regularizing the policy with a direct KL divergence term (rather than a per-token KL penalty in the reward) is conceptually cleaner and avoids entangling the reward model's output with the regularization term. This separation could improve interpretability and allow for more precise tuning of the KL coefficient.

5. **Process Supervision Capability**:
   - Your algorithm is flexible enough to incorporate process supervision, where intermediate reasoning steps are scored. This could be particularly beneficial for mathematical reasoning tasks, where the quality of intermediate steps is crucial.

---

### **Weaknesses**

1. **Bias in Advantage Estimation**:
   - While using group-relative rewards as advantages simplifies computation, it introduces bias because the baseline (group mean) depends on the sampled outputs. This baseline is not an unbiased estimate of the expected return, which could lead to suboptimal policy updates.

2. **High Variance**:
   - Using rewards normalized within a small group of outputs can lead to high variance in the advantage estimates, especially if the rewards within the group are noisy or if the group size \(G\) is small. This could destabilize training.

3. **Lack of Temporal Credit Assignment**:
   - Without a value function, your method lacks a principled way to perform temporal credit assignment. In tasks where intermediate tokens contribute differently to the final reward, this could lead to suboptimal learning. While process supervision partially addresses this, it relies on having a reliable process reward model, which may not always be available.

4. **Dependence on Group Sampling**:
   - The quality of the policy updates depends on the diversity and representativeness of the sampled group of outputs. If the group is not representative of the policy's overall behavior, the updates could be biased or inefficient.

5. **KL Divergence Estimation**:
   - Your KL divergence estimator, while unbiased, involves ratios of probabilities, which can be unstable if the policy assigns very low probabilities to certain tokens. This could lead to numerical instability or exploding gradients.

6. **Scalability to Large Models**:
   - While your method avoids the overhead of a value function, it requires sampling multiple outputs per question (\(G\)), which increases the computational cost. For large LLMs, this could offset some of the savings from not training a value function.

---

### **Comparison to PPO with GAE**

1. **Bias vs. Variance**:
   - PPO with GAE uses a learned value function to estimate advantages, which introduces bias due to imperfect value function approximation but reduces variance through bootstrapping. Your method eliminates this bias but suffers from higher variance due to group-relative normalization.
   - In practice, the higher variance in your method could lead to slower convergence or instability, especially if \(G\) is small or the reward model is noisy.

2. **Credit Assignment**:
   - PPO with GAE can perform temporal credit assignment through multi-step returns and bootstrapping. Your method relies on process supervision for this, which may not always be feasible or accurate.

3. **Stability**:
   - PPO with GAE benefits from the value function's stabilizing effect on advantage estimation. Without this, your method may require careful tuning of hyperparameters (e.g., learning rate, \(G\), and \(\epsilon\)) to ensure stable training.

4. **Computational Cost**:
   - Your method avoids the cost of training a value function but requires sampling multiple outputs per question. The trade-off between these costs depends on the size of the model and the group size \(G\).

5. **Alignment with Comparative Rewards**:
   - Your group-relative advantage calculation is better aligned with the comparative nature of reward models, which could lead to more effective updates in some cases.

---

### **Theoretical Concerns**

1. **Bias in Baseline**:
   - The group mean reward is not an unbiased estimate of the expected return, which introduces bias in the policy updates.

2. **High Variance**:
   - Normalizing rewards within a group can amplify noise, leading to high variance in the updates. This could require larger batch sizes or more aggressive clipping (\(\epsilon\)) to stabilize training.

3. **Numerical Instability**:
   - The KL divergence estimator involves ratios of probabilities, which can be unstable if the policy assigns very low probabilities to certain tokens.

4. **Overfitting to Group**:
   - The policy may overfit to the specific sampled group of outputs, leading to suboptimal generalization.

---

### **Practical Concerns**

1. **Reward Model Quality**:
   - The effectiveness of your method depends heavily on the quality of the reward model. If the reward model is noisy or misaligned, the group-relative normalization could amplify these issues.

2. **Hyperparameter Sensitivity**:
   - Your method introduces additional hyperparameters (\(G\), \(\beta\), etc.) that need to be carefully tuned for stability and performance.

3. **Scalability**:
   - Sampling multiple outputs per question increases computational cost, which could be prohibitive for large LLMs.

4. **Process Supervision Reliance**:
   - While process supervision is a strength, it requires reliable intermediate step rewards, which may not always be available or easy to define.

---

### **Expected Performance**

1. **Outperformance**:
   - Your method could outperform PPO in settings where the reward model is highly reliable, group sampling is diverse, and process supervision is available. The alignment with comparative rewards could lead to more effective updates.

2. **Matching PPO More Cheaply**:
   - If the value function overhead is a bottleneck, your method could match PPO's performance at a lower cost, provided the variance is well-controlled.

3. **Underperformance**:
   - In settings with high reward noise, sparse rewards, or poor group sampling, your method is likely to underperform PPO due to high variance and lack of temporal credit assignment.

---

### **Recommendations**

1. **Experiment with Group Size**:
   - Carefully tune \(G\) to balance computational cost and variance reduction.

2. **Variance Reduction Techniques**:
   - Consider additional variance reduction techniques, such as reward normalization across larger batches or using a learned baseline.

3. **Hybrid Approach**:
   - Explore hybrid approaches that combine your group-relative rewards with a lightweight value function for better temporal credit assignment.

4. **Robust KL Estimation**:
   - Use regularization or clipping to stabilize the KL divergence estimation.

5. **Empirical Validation**:
   - Compare your method to PPO with GAE on a range of tasks to empirically evaluate its performance and stability.

In summary, your algorithm is promising but comes with trade-offs in bias, variance, and scalability. Its success will depend on careful tuning and the specific characteristics of the task and reward model.
~~~~
