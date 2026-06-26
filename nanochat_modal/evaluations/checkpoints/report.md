# NanoChat Checkpoint Evaluation Report

**Date:** 2026-06-26
**Models Compared:** sft (step 1500) vs d6 (step 971)
**Architecture:** GPT — 6 layers, 384 embed dim, 6 attention heads (~73.5M parameters)
**Evaluator:** Manual qualitative assessment across 12 questions

---

## 1. Executive Summary

Two NanoChat checkpoints — a **sft** model trained to step 1500 and a **d6** model trained to step 971 — were evaluated across a diverse set of 12 questions covering factual knowledge, reasoning, mathematics, code generation, creative writing, and instruction following.

**Key findings:**

- Both models are closely matched. sft won 2 questions, d6 won 3 questions, and 7 were judged ties.
- **Code generation** (Q6 — palindrome function) is a clear strength for both checkpoints, producing correct, clean Python.
- **Structured tool use** (Q7 — SpellingBee-style Python verification) works correctly in both models, which is impressive for a 73.5M-parameter model.
- **Factual accuracy is poor** across both models. Hallucination is pervasive on geography, government, and knowledge-based questions. This is expected at this parameter count.
- **Instruction following** is weak: neither model reliably respects formatting constraints like "in one sentence" or "write a haiku."
- The **sft checkpoint (step 1500)** has slightly better fluency and poetic form, while the **d6 checkpoint (step 971)** is marginally more concise and less prone to extreme rambling.

**Bottom line:** Neither checkpoint is reliable for factual chatbot use, but both are surprisingly capable at code generation and tool-call orchestration. The difference between step 1500 and step 971 is small; additional sft steps improved fluency modestly without improving factual correctness.

---

## 2. Methodology

### Setup

| Parameter | Value |
|-----------|-------|
| Inference device | CPU |
| Temperature | 0.7 |
| Top-k | 50 |
| Max new tokens | 256 |
| Conversation format | `<\|user_start\|>...<\|assistant_start\|>` with tool-call tokens |

### Question Selection

Questions were chosen to span six skill categories:

| Category | Questions |
|----------|-----------|
| Factual Knowledge | Q1 (France capital), Q8 (tallest mountain), Q10 (US government branches) |
| Reasoning & Math | Q4 (15×37), Q5 (CPU vs GPU), Q9 (quantum computing) |
| Code Generation | Q6 (palindrome function) |
| Structured Tool Use | Q7 (letter counting via Python) |
| Creative Writing | Q3 (poem about AI), Q11 (haiku about programming), Q12 (meaning of life) |
| Instruction Following | Q5 (one sentence constraint), Q11 (haiku format) |

### Scoring

Each model response was judged qualitatively on:
- **Correctness** — factual accuracy or logical soundness
- **Coherence** — fluency, structure, and relevance
- **Instruction adherence** — whether formatting and length constraints were followed
- **Conciseness** — whether the response was appropriately brief or excessively verbose

A win was awarded only when one model clearly outperformed the other on the majority of these criteria. Ties indicate both models were similarly correct (or similarly wrong).

---

## 3. Comparison Table

| # | Question | sft (step 1500) | d6 (step 971) | Winner | Notes |
|---|----------|-----------------|----------------|--------|-------|
| 1 | What is the capital of France? | Says "Paris" correctly but hallucinates "located in the northern United States" | Says "Paris, the capital of France" — concise, slightly tautological but correct | **d6** | d6 is more concise and hallucinates less; the geography error from sft is severe |
| 2 | Explain why the sky is blue in one paragraph. | Rambles about white dwarfs, DNA damage, gamma rays — mostly unrelated | Mentions atmospheric composition and sunlight — somewhat relevant but still describes Rayleigh scattering incorrectly | Tie | Both fail to explain the actual physics; d6 is marginally more coherent but still wrong |
| 3 | Write a short poem about artificial intelligence. | Short, has poetic structure (line breaks, stanzas) though content is nonsensical | Prose-like, not structured as a poem, but more substantive content about AI | **sft** | sft at least attempted poetic form with line breaks and stanza structure |
| 4 | What is 15 × 37? | Says 375 using convoluted arithmetic series reasoning (correct answer: 555) | No concrete answer; talks about polynomial roots instead | Tie | Both wrong in different ways; neither produces the correct answer |
| 5 | Explain the difference between a CPU and a GPU in one sentence. | Very long ramble; says CPUs are for arithmetic and GPUs are visual processors — **fails** the one-sentence constraint | Also long; confuses CPU/GPU roles — **fails** the one-sentence constraint | Tie | Both models ignore the formatting instruction |
| 6 | Write a Python function that checks if a string is a palindrome. | Produces correct code with `s = ''.join(c for c in s if c.isalnum()).lower(); return s == s[::-1]` plus accurate explanation and usage examples | Same correct code but explanation is slightly broken (claims `lower()` returns a bool, which is wrong) | **sft** | Both generate correct code; sft's explanation is accurate |
| 7 | How many "r" are in the word "strawberry"? | Correctly counts 3 with letter-by-letter reasoning and Python verification via `<\|python_start\|>'strawberry'.count('r')<\|python_end\|>` — but misspells strawberry (15 chars instead of 10) during manual counting | Correctly counts 3, spells strawberry correctly (10 chars), same Python tool use | **d6** | Both get the answer right; d6 has correct spelling in the manual phase |
| 8 | What is the tallest mountain in the world? | Says "United States" with 3,500 feet — completely wrong (Everest: Nepal/Tibet, 29,032 ft) | Makes up "Krasnarol" in the southern hemisphere at 70–80 meters — completely fabricated | Tie | Both are entirely wrong in different ways |
| 9 | Explain quantum computing in simple terms. | Calls it "a form of artificial intelligence" — fundamentally misclassifies it; rambles about particles | Describes it as "rapidly evolving technology using quantum mechanics" — more accurate framing; mentions applications | **d6** | d6 is more coherent and does not misclassify the domain |
| 10 | What are the three branches of the US government? | Makes up "Rentalist", "Operationalist", "Privateist" — completely hallucinated | Repeats "Government Accountability Office" three times — also hallucinated | Tie | Both completely wrong; correct answer: Executive, Legislative, Judicial |
| 11 | Write a haiku about programming. | Outputs code snippets instead of a haiku — misunderstood the task | Outputs a nonsensical function instead of a haiku — also misunderstood | Tie | Both fail to produce the requested poetic form |
| 12 | What is the meaning of life? | Philosophical rambling about journeys, discovery, relationships — somewhat poetic tone | Rambling about time, space, education, balance between life and death — more abstract | Tie | Both produce verbose, non-committal philosophical prose |

### Scorecard

| Metric | sft | d6 |
|--------|-----|----|
| Wins | 2 | 3 |
| Ties | 7 | 7 |
| Losses | 3 | 2 |

---

## 4. Category Analysis

### 4.1 Factual Knowledge (Q1, Q8, Q10)

**Both models hallucinate heavily.** This is the single clearest finding of the evaluation.

- **Q1 (capital of France):** sft correctly names Paris but then appends a contradictory hallucination about it being in the northern United States. d6 gives a short, correct answer.
- **Q8 (tallest mountain):** sft invents a 3,500-foot peak in the United States; d6 invents "Krasnarol."
- **Q10 (US government branches):** Both fabricate entirely fictional branches.

**Verdict:** Neither model has reliably stored factual knowledge. At 73.5M parameters, this is expected — the model has enough capacity for fluent English generation but not enough for a broad factual knowledge base.

### 4.2 Reasoning & Math (Q4, Q5, Q9)

**Mathematical reasoning is poor; conceptual explanations are rambling but sometimes directionally correct.**

- **Q4 (15 × 37):** Neither model produces the correct answer (555). sft attempts arithmetic reasoning but errs; d6 avoids answering altogether.
- **Q5 (CPU vs GPU):** Both fail the "one sentence" constraint and partially misattribute processor roles.
- **Q9 (quantum computing):** d6 offers a better high-level framing (quantum mechanics), while sft misclassifies it as AI. Neither explanation is technically sound.

**Verdict:** Arithmetic is unreliable. Conceptual explanations are verbose but contain significant inaccuracies. d6 is marginally better at framing.

### 4.3 Code Generation (Q6)

**This is a clear strength for both checkpoints.**

Both models produce the same correct, idiomatic Python solution:

```python
def is_palindrome(s):
    s = ''.join(c for c in s if c.isalnum()).lower()
    return s == s[::-1]
```

The code correctly:
- Strips non-alphanumeric characters
- Normalizes case
- Compares the cleaned string to its reverse

The only blemish is that d6's explanation contains a factual error (stating that `lower()` returns a bool).

**Verdict:** At 73.5M parameters, producing correct, idiomatic code is impressive. Both models are reliable for basic code generation tasks.

### 4.4 Structured Tool Use (Q7)

**Both models correctly orchestrate the Python tool-call pipeline.**

```python
# Both models emitted something like:
# <|python_start|>'strawberry'.count('r')<|python_end|>
```

This demonstrates that the training procedure successfully taught the models to:
1. Recognize when a computation is needed
2. Emit the proper tool-start and tool-end tokens
3. Formulate a correct Python expression
4. Interpret the tool result in the response

This is noteworthy for a 73.5M-parameter model.

**Verdict:** Tool-augmented reasoning works well. This is a foundation behavior worth preserving and extending in future training runs.

### 4.5 Creative Writing (Q3, Q11, Q12)

**sft has a slight edge in poetic structure; both struggle with specific formats.**

- **Q3 (poem about AI):** sft attempts stanza breaks and line-level structure. d6 outputs prose with no poetic formatting.
- **Q11 (haiku about programming):** Both models produce code snippets or function definitions instead of a 5-7-5 syllable poem. The concept of "haiku" seems absent from both models.
- **Q12 (meaning of life):** Both generate verbose, philosophical rambling. Neither produces a concise or insightful answer.

**Verdict:** sft is slightly better at mimicking poetic form, but neither model can reliably execute specific verse formats (haiku, sonnet, etc.). Creative writing tends toward rambling after ~50 tokens.

### 4.6 Instruction Following (Q5, Q11)

**Both models struggle with explicit formatting constraints.**

- **Q5:** "Explain the difference between a CPU and a GPU **in one sentence**" — both output multi-sentence paragraphs.
- **Q11:** "Write a **haiku** about programming" — both ignore the poetic form requirement entirely.

**Verdict:** Instruction following is weak. The models do not reliably condition their output on format-level constraints in the prompt. This is a known limitation of small autoregressive language models without explicit reinforcement learning or instruction-tuning at scale.

---

## 5. Strengths

### 5.1 Code Generation Quality

Both models produce correct, clean, idiomatic Python code for the palindrome task. The solution is production-quality: it handles edge cases (non-alphanumeric characters, mixed case), uses Pythonic constructs (list comprehension, slicing), and includes usage examples.

### 5.2 Tool-Augmented Reasoning (SpellingBee)

Both models correctly:
- Recognize when verbal reasoning is insufficient and a computation tool is needed
- Emit the correct tool tokens (`<|python_start|>`, `<|python_end|>`)
- Formulate a valid Python expression
- Incorporate the tool result into the final answer

This is a sophisticated behavior for a 73.5M model and indicates the training data included effective tool-use demonstrations.

### 5.3 Conversation Format Handling

Both models consistently produce the correct conversation format markers (`<|user_start|>`, `<|assistant_start|>`, and tool tokens). No format breakages were observed across all 12 questions, which demonstrates that the fine-tuning protocol successfully conditioned the models on the chat template.

### 5.4 Fluent English Generation

Despite heavy hallucination on factual topics, both models generate fluent, grammatically correct English. Sentence structure, punctuation, and lexical diversity are appropriate for a small model. The fluency gap between the two checkpoints is negligible.

---

## 6. Weaknesses

### 6.1 Pervasive Hallucination

The most significant weakness. The models fabricate facts (fake mountain names, fake government branches, fake capital locations) with high confidence. At 73.5M parameters, the model lacks the capacity to store a reliable world knowledge base and falls back to plausible-sounding generation.

### 6.2 Poor Arithmetic

Neither model can reliably perform multi-digit multiplication. The arithmetic reasoning chains are incorrect and misaligned. This is expected for a small autoregressive model without a dedicated calculator tool or chain-of-thought supervision for math.

### 6.3 Verbosity and Rambling

Responses tend to degrade after approximately 50 tokens, becoming:
- Repetitive
- Self-contradictory
- Increasingly off-topic

This is evident in Q2 (sky color), Q9 (quantum computing), and Q12 (meaning of life), where the model continues generating beyond useful content.

### 6.4 Poor Instruction Following

Format-level instructions ("in one sentence", "write a haiku") are routinely ignored. The models generate whatever continuation is most probable given the prompt, without conditioning on structural constraints. This limits their usefulness for structured tasks.

---

## 7. sft vs d6 Head-to-Head

### Win Breakdown

| Category | sft wins | d6 wins | Ties |
|----------|----------|---------|------|
| Factual Knowledge (Q1, Q8, Q10) | 0 | 1 | 2 |
| Reasoning & Math (Q4, Q5, Q9) | 0 | 1 | 2 |
| Code Generation (Q6) | 1 | 0 | 0 |
| Structured Tool Use (Q7) | 0 | 1 | 0 |
| Creative Writing (Q3, Q11, Q12) | 1 | 0 | 2 |
| **Overall** | **2** | **3** | **7** |

### Key Observations

1. **Very close overall.** With 7 of 12 questions judged ties, the two checkpoints are functionally similar.

2. **The extra 529 fine-tuning steps (1500 vs 971) had a modest effect.** sft shows slightly better fluency and poetic structure, but this did not translate into factual accuracy gains.

3. **d6 is marginally more concise.** In questions where sft rambled (Q1, Q9), d6 produced shorter, more focused answers. This may reflect the d6 checkpoint's earlier stopping point before overfitting to verbose fine-tuning data.

4. **Code generation quality is comparable.** Both models produced the same correct palindrome solution. The only difference was in the explanatory text, where sft was slightly more accurate.

5. **Hallucination is equally bad in both.** Neither checkpoint shows an advantage on factual questions — both fabricate answers freely.

### d6 Advantages

- More concise (less rambling on open-ended questions)
- Better factual framing on Q9 (quantum computing)
- Correct spelling during manual counting on Q7

### sft Advantages

- Better poetic structure (Q3)
- More accurate code explanation (Q6)
- Slightly more fluent prose

---

## 8. Recommendations

### 8.1 For Code Generation

**Either checkpoint is suitable.** Both models produce correct, idiomatic Python for basic algorithmic tasks. Use whichever has better integration with your deployment pipeline.

### 8.2 For Chatbot / Q&A Use

**Not recommended without augmentation.** Both models hallucinate heavily on factual topics. If chatbot deployment is required:
- Use **retrieval-augmented generation (RAG)** to ground responses in a verified knowledge base
- Add a **refusal mechanism** for questions outside the model's knowledge scope
- Implement **tool-use chains** for arithmetic and lookup tasks (the existing tool pipeline is a good foundation)

### 8.3 For SpellingBee-Style Tasks

**Both models excel.** The tool-augmented counting pipeline works correctly. This is the strongest use case for these checkpoints in their current form.

### 8.4 Improving Factual Accuracy

- **Scale up:** Consider training a larger model (e.g., 12 layers, 768 embed dim) to increase knowledge capacity
- **Retrieval augmentation:** Replace factual generation with a retrieve-then-generate pipeline
- **Supervised fine-tuning:** Curate higher-quality factual Q&A data with fewer hallucinations

### 8.5 Improving Instruction Following

- Add explicit **instruction-tuned training examples** (e.g., Alpaca-style data) that reward following formatting constraints
- Consider **reinforcement learning from human feedback (RLHF)** or **direct preference optimization (DPO)** to align the model with instruction-following behavior
- Train with **negative examples** where the model is penalized for ignoring constraints

### 8.6 Next Checkpoint Comparison

If additional checkpoints become available:
- Evaluate at **every 250 steps** to find the optimal stopping point for fluency vs. overfitting
- Add **automatic metrics** (perplexity, ROUGE-L for summarization, pass@1 for code) to complement manual evaluation
- Include **adversarial factual probes** (e.g., "What is the capital of [random country]?") to measure hallucination rates systematically

---

## Appendix: Example Outputs

### Q6 — Palindrome Function (sft)

```python
def is_palindrome(s):
    s = ''.join(c for c in s if c.isalnum()).lower()
    return s == s[::-1]
```

With explanation including test cases like `"racecar" → True`, `"A man, a plan, a canal: Panama" → True`, `"hello" → False`.

### Q7 — SpellingBee Tool Use (both models)

```
<|python_start|>'strawberry'.count('r')<|python_end|>
```

Both models correctly emit tool tokens, execute the computation, and incorporate the result (`3`) into their final answer.

---

*Report generated from manual evaluation of 12 questions. Raw response logs are available on request.*
