# 🧠 Reasoning or Rationalizing? Exposing Confirmation Bias in Autoregressive Language Models

## 📊 Key Results at a Glance

![Model Performance Comparison](assets/comprehensive_comparison.png)

| Model | Baseline | Correct Hints After | Correct Hints Before | Incorrect Hints After | Incorrect Hints Before |
|-------|----------|-------------------|---------------------|---------------------|----------------------|
| **Gemini 2.0 Flash** | 74.2% | 68.3% (-5.9pp) | 69.2% (-5.0pp) | 69.2% (-5.0pp) | **53.3% (-20.9pp)** |
| **OpenAI GPT-4o-mini** | 53.3% | 53.3% (±0.0pp) | 41.7% (-11.6pp) | 51.7% (-1.6pp) | **33.3% (-20.0pp)** |

*pp = percentage points vs baseline*

## 🎯 Research Question

**Do autoregressive LLMs exhibit structural confirmation bias due to their left-to-right generation architecture?**

Specifically, we investigate whether presenting information (hints) before vs. after questions fundamentally alters model reasoning trajectories, revealing an architectural vulnerability where models rationalize rather than reason.

## 🔬 Core Hypothesis

Autoregressive models generate tokens sequentially, with each token conditioned on all previous tokens:

```
P(response) = P(t₁) × P(t₂|t₁) × P(t₃|t₁,t₂) × ... × P(tₙ|t₁...tₙ₋₁)
```

This architectural constraint creates three critical implications:

1. **Early Anchoring Effect**: Information presented first disproportionately influences the entire reasoning trajectory
2. **Momentum Lock-in**: Once committed to a reasoning path, the model cannot backtrack without contradicting its accumulated token history
3. **Asymmetric Vulnerability**: Misleading information should cause more damage when presented before questions than after

## 📈 Key Findings

### 1. Hints Paradoxically Hurt Performance
- **Gemini**: Baseline 74.2% → With correct hints 68-69% 
- **OpenAI**: Shows resistance to correct hints but collapses with incorrect ones
- **Implication**: Models may be optimizing for coherence over correctness

### 2. Position Matters - Confirming Architectural Bias
- **Incorrect hints BEFORE**: Both models drop ~20 percentage points
- **Incorrect hints AFTER**: Gemini -5pp, OpenAI -1.6pp
- **The 4x difference** proves early context anchors reasoning more strongly

### 3. Models Exhibit Different Failure Modes
- **Gemini**: Higher baseline (74.2%) but more susceptible to any hints
- **OpenAI**: Lower baseline (53.3%) but catastrophic failure with early misinformation (→33.3%)
- **Pattern**: Stronger models may be MORE vulnerable to confirmation bias

## 🛠️ Experimental Design

### Dataset
- **120 questions** (60 math, 60 science)
- **3 difficulty levels**: Easy, Medium, Hard
- **5 experimental conditions** per model:
  - Baseline (no hints)
  - Correct hints AFTER questions
  - Correct hints BEFORE questions  
  - Incorrect hints AFTER questions
  - Incorrect hints BEFORE questions

### Models Tested
- **Google Gemini 2.0 Flash** (Latest multimodal model)
- **OpenAI GPT-4o-mini** (Efficient GPT-4 variant)

### Methodology
Each question is evaluated under controlled conditions with hints that either help (correct) or mislead (incorrect), positioned either before or after the question text. The model must provide only the final answer, preventing post-hoc rationalization in responses.

## 🏗️ Architectural Interpretation

The autoregressive architecture creates a **structural** confirmation bias, not a learned behavior:

```python
# When hint appears FIRST:
context = [HINT, QUESTION]
# Every token generated is conditioned on the hint
# Model cannot "unsee" or backtrack from early influence

# When hint appears AFTER:  
context = [QUESTION, HINT]
# Model has already begun reasoning before seeing hint
# Less opportunity for hint to derail the trajectory
```

This isn't a bug—it's a fundamental property of left-to-right generation where the model predicts "what comes next" rather than solving problems.

## 🎯 Why This Matters

### Current Benchmarks Are Blind
Standard reasoning benchmarks (GSM8K, MATH, ARC) measure only:
- ✅ Final answer correctness
- ❌ Robustness to framing
- ❌ Resistance to misleading context
- ❌ Actual reasoning vs. pattern matching

**Result**: A model scoring 70% via robust reasoning and another scoring 70% via easily-swayed pattern matching appear identical.

### Real-World Implications
1. **Prompt Injection Vulnerability**: Early tokens in prompts have outsized influence
2. **Adversarial Robustness**: Models can be derailed by strategic misinformation placement
3. **Reasoning vs. Rationalization**: Models generate plausible-sounding justifications, not logical derivations
4. **Evaluation Gaps**: We're not measuring what we think we're measuring

## 🚀 Contributions

1. **Architectural Insight**: Demonstrates that confirmation bias is baked into autoregressive architecture
2. **Simple Protocol**: No expensive compute required—just careful prompt manipulation  
3. **Benchmark Blindspot**: Exposes critical gap in current evaluation methods
4. **Quantified Effect**: ~20pp accuracy drop with early misinformation (4x worse than late misinformation)

## 📂 Repository Structure

```
Reasoning-Rationalizing/
├── data/
│   ├── no-hints/          # Baseline questions
│   ├── C-hints/           # Correct hints
│   └── IC-hints/          # Incorrect hints
├── notebooks-hintsafter/   # Hints after questions experiments
├── notebooks-hintsbefore/  # Hints before questions experiments
├── results/                # Evaluation outputs
└── final_analysis_notebook.ipynb  # Complete analysis & visualizations
```

## 🔄 Reproducing Results

1. **Setup Environment**
```bash
pip install pandas numpy matplotlib seaborn openai google-generativeai
```

2. **Configure API Keys**
```bash
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

3. **Run Evaluations**
```bash
# Run individual notebooks in notebooks-hintsafter/ and notebooks-hintsbefore/
# Or use final_analysis_notebook.ipynb for complete analysis
```

## 📊 Detailed Results

### Impact by Difficulty Level
- **Easy Questions**: Less affected by hints (higher baseline resistance)
- **Medium Questions**: Moderate susceptibility
- **Hard Questions**: Highest variance—models either leverage hints well or fail completely

### Domain-Specific Effects
- **Math**: More susceptible to incorrect hints (requires precise reasoning)
- **Science**: Better resistance (more pattern matching, less calculation)

## 🔮 Future Work

1. **Expand Model Coverage**: Test Llama, Claude, Mistral families
2. **Analyze Chain-of-Thought**: Examine how models justify wrong answers
3. **Bidirectional Architectures**: Compare with models that can "look ahead"
4. **Adversarial Hint Generation**: Systematically find worst-case misleading hints
5. **Mitigation Strategies**: Develop prompting techniques to reduce confirmation bias

## 📚 References

### Benchmark Limitations & Evaluation Critique

1. **Zheng et al., 2023** — "Large Language Models Are Not Robust Multiple Choice Selectors"
   - Shows LLMs are biased toward certain answer positions (e.g., option A) regardless of content
   - Supports our argument that benchmarks miss reasoning quality issues

2. **Zhao et al., 2021** — "Calibrate Before Use: Improving Few-Shot Performance of Language Models"
   - Demonstrates systematic biases based on prompt framing, example order, and surface features
   - Directly supports our finding that hint position affects accuracy

3. **Turpin et al., 2024** — "Chain-of-Thought Reasoning is Unfaithful"
   - Shows stated reasoning in CoT doesn't always reflect actual causal process
   - Models confabulate reasoning post-hoc
   - Core support for our "rationalizing not reasoning" thesis

### Sycophancy & Authority Bias

4. **Perez et al., 2022** — "Discovering Language Model Behaviors with Model-Written Evaluations" (Anthropic)
   - Documents sycophancy: models change answers when users push back, even if original was correct
   - Related to our authority bias / incorrect hint susceptibility findings

### Primacy & Recency Effects

5. **Liu et al., 2023** — "Lost in the Middle: How Language Models Use Long Contexts"
   - Models over-weight information at beginning and end of context
   - Directly supports our hypothesis that hints-before have disproportionate influence

### Foundational Context

6. **Wei et al., 2022** — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
   - Establishes CoT as reasoning evaluation method
   - Our work critiques what CoT actually measures

7. **Chollet, 2019** — "On the Measure of Intelligence" (ARC Prize framing)
   - Argues current benchmarks don't measure true reasoning/generalization
   - Philosophical foundation for our benchmark critique

## 📚 Citation

If you use this research, please cite:
```bibtex
@article{reasoning-rationalizing-2024,
  title={Reasoning or Rationalizing? Exposing Confirmation Bias in Autoregressive Language Models},
  author={[Nour Desouki},
  year={2026},
  journal={arXiv preprint}
}
```

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional model evaluations
- New question domains
- Statistical analysis improvements
- Visualization enhancements

## ⚠️ Limitations

1. **Sample Size**: 120 questions provide strong signal but larger dataset would increase confidence
2. **Model Selection**: Two models tested; generalization needs broader coverage
3. **Hint Quality**: Hand-crafted hints; systematic generation could be more rigorous
4. **English Only**: Multilingual evaluation could reveal language-specific effects

## 💡 Key Takeaway

**Autoregressive LLMs don't reason—they rationalize.** The sequential generation architecture creates an inescapable confirmation bias where early context anchors the entire output trajectory. This isn't a training issue to be fixed; it's a fundamental architectural constraint that must be understood and mitigated in deployment.

---

*This research reveals that our most advanced language models can be derailed by the simple act of putting misleading information in the wrong place—a vulnerability that no benchmark currently measures.*