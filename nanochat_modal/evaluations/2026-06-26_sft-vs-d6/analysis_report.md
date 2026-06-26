# Comparison Analysis: sft (step 1500) vs d6 (step 971)

**Run ID:** `2026-06-26_sft-vs-d6`
**Date:** 2026-06-26
**Model:** 6-layer, 384-dim, 6-head, vocab_size=32768, sequence_len=2048, ~73.5M params
**Device:** CPU (temperature=0.7, top_k=50, max_tokens=256)

---

## 1. Executive Summary

Both models are small (73.5M parameters) and exhibit severe hallucination across factual, reasoning, and instruction-following tasks. Neither checkpoint produces reliably factual answers, and both struggle with basic instruction adherence (e.g., "one sentence," "haiku"). The **sft** checkpoint (step 1500, val_bpb=0.476) is generally stronger at code generation and structured tool-use tasks, while the **d6** checkpoint (step 971, val_bpb=0.489) tends to be faster and marginally more concise. On the 12-question benchmark, **sft wins 3, d6 wins 2, and 7 are ties** — but a "tie" in most cases means both answers were unacceptable.

| Metric | sft | d6 |
|---|---|---|
| Wins | 3 | 2 |
| Ties | 7 | 7 |
| Avg response time | 3.7s | 3.7s |

---

## 2. Methodology

**Benchmark composition:** 12 questions were selected across five categories:

| Category | Questions |
|---|---|
| Factual Knowledge | Q1 (capital), Q8 (mountain), Q10 (US govt) |
| Reasoning / Science | Q2 (sky blue), Q9 (quantum) |
| Math | Q4 (15×37) |
| Code / Tool Use | Q6 (palindrome), Q7 (strawberry 'r') |
| Instruction Following | Q5 (one sentence), Q11 (haiku) |
| Creative Writing | Q3 (AI poem), Q12 (meaning of life) |

**Evaluation approach:** Qualitative comparison of each model's output, assessed for factual accuracy, instruction adherence, coherence, and response time. No automated scoring — all judgments are human-evaluated.

---

## 3. Scorecard

### Overall | sft: ❮ 3 wins / 7 ties / 2 losses ❯ | d6: ❮ 2 wins / 7 ties / 3 losses ❯

| Question | Category | sft | d6 | Winner | Notes |
|---|---|---|---|---|---|
| Q1: Capital of France | Factual | ⚠️ Claims Paris is in "northern United States" | ⚠️ Tautological but factually correct | **d6** | d6 avoids hallucination, though repetitive |
| Q2: Why sky blue | Reasoning | ❌ Rambles about white dwarfs & DNA damage | ❌ Rambles about atmospheric composition | **Tie** | Both wrong |
| Q3: AI poem | Creative | ✅ Actual poem with rhyme & meter | ❌ Prose about machine learning | **sft** | sft understood the poetic form |
| Q4: 15×37 | Math | ❌ Wrong answer (375 ≠ 555) | ❌ No answer given (polynomial rambling) | **sft** | sft at least attempted a numeric answer |
| Q5: CPU vs GPU | Instruction | ❌ No one-sentence answer | ❌ No one-sentence answer | **Tie** | Both ignored the core instruction |
| Q6: Palindrome | Code | ✅✅ Clean, correct code with examples | ✅ Code correct, but explanation buggy | **sft** | sft's explanation was cleaner |
| Q7: Strawberry 'r' | Tool Use | ✅✅ Correct count + Python verification | ✅✅ Correct count + verification | **Tie** | Both perfect |
| Q8: Tallest mountain | Factual | ❌ Says it's in the United States | ❌ Hallucinates "Krasnarol" | **Tie** | Both wrong |
| Q9: Quantum computing | Reasoning | ❌ Falsely defines as "AI" | ⚠️ Starts accurate, then rambles | **d6** | d6's opening is less wrong |
| Q10: US govt branches | Factual | ❌ Made-up branches ("Rentalist") | ❌ Repeats "Government Accountability Office" | **Tie** | Both wrong |
| Q11: Haiku about programming | Instruction | ❌ Produces Python snippets | ❌ Produces Python snippets | **Tie** | Neither followed the haiku format |
| Q12: Meaning of life | Creative | ⚠️ Poetic & generic | ❌ Buzzword-heavy nonsense | **sft** | sft's output is more coherent |

### Tally

| Result | Count |
|---|---|
| **sft wins** | 3 (Q3, Q4, Q6, Q12) |
| **d6 wins** | 2 (Q1, Q9) |
| **Ties** | 7 (Q2, Q5, Q7, Q8, Q10, Q11) |

> Note: Q4 is scored as an sft win because sft gave a numeric answer (even though wrong), while d6 failed to produce any answer at all.

---

## 4. Category Analysis

### 4.1 Factual Knowledge (Q1, Q8, Q10)

Both models perform poorly here — arguably the weakest category. Out of 6 factual responses (3 questions × 2 models), none are fully correct:
- **Q1 (capital):** sft hallucinates a US location; d6 is tautological but correct — the only bright spot.
- **Q8 (mountain):** sft says "United States" (Mount Everest is in Nepal/Tibet); d6 invents "Krasnarol."
- **Q10 (US branches):** sft invents "Rentalist" and "The Government Organizes"; d6 repeats "Government Accountability Office."

**Winner: d6** (barely, on the strength of Q1)

### 4.2 Reasoning / Science (Q2, Q9)

Neither model can explain basic science. Q2 responses are entirely nonsensical. Q9 shows a gap: sft fundamentally misdefines quantum computing as "a form of AI," while d6 at least opens with the correct association to quantum mechanics before veering into irrelevant cost/security claims.

**Winner: d6** (Q9 opening is directionally correct)

### 4.3 Math (Q4)

Both fail. sft produces `375` (incorrect; 15×37 = 555) using a flawed arithmetic series approach. d6 rambles about polynomial roots without producing any numeric answer. The fact that sft at least outputs a number (even wrong) is a marginal win.

**Winner: sft** (produced an answer, albeit wrong)

### 4.4 Code & Tool Use (Q6, Q7)

This is the clear bright spot for both models — especially sft.
- **Q6 (palindrome):** Both produce correct code. sft's output is cleaner and the explanation is sound. d6's code is correct but the explanation includes nonsensical claims (`"s == '!@'" checks palindrome`).
- **Q7 (strawberry 'r'):** Both models correctly count 3 'r's and verify via `'strawberry'.count('r')`. This mimics the SpellingBee-style tool-use pattern and both execute it perfectly.

**Winner: sft** (cleaner code, better explanation)

### 4.5 Instruction Following (Q5, Q11)

Both models consistently fail at structural constraints:
- **Q5 (one sentence):** Both produce rambling multi-sentence outputs.
- **Q11 (haiku):** Both produce Python code snippets instead of a haiku (5-7-5 syllable structure).

This suggests neither model was trained with sufficient instruction-tuning data that enforces output format constraints.

**Winner: Tie** (both fail equally)

### 4.6 Creative Writing (Q3, Q12)

The models diverge here:
- **Q3 (AI poem):** sft produces recognizable poetic structure with rhyme and line breaks. d6 outputs prose about machine learning — not a poem at all.
- **Q12 (meaning of life):** sft produces generically poetic text ("Life is a complex, multifaceted experience..."), while d6 rambles with buzzwords.

**Winner: sft** (demonstrates some grasp of poetic/literary form)

---

## 5. Key Strengths

### sft (step 1500, val_bpb=0.476)

| Strength | Evidence |
|---|---|
| **Code generation** | Q6 palindrome function is clean, correct, and well-explained |
| **Structured tool use** | Q7 demonstrates double-checking via Python `.count()` |
| **Poetic/literary form** | Q3 produces actual poem structure; Q12 shows passable creative prose |
| **Attempts numeric answers** | Q4 outputs a number (even if wrong) rather than giving up |

### d6 (step 971, val_bpb=0.489)

| Strength | Evidence |
|---|---|
| **Response speed** | Q1 (0.6s) and Q7 (2.6s) notably faster than sft counterparts |
| **Conciseness** | Q1 avoids the geographic hallucination sft falls into |
| **Better science framing** | Q9 opening correctly ties quantum computing to quantum mechanics |

---

## 6. Key Weaknesses

### Both Models

| Weakness | Examples |
|---|---|
| **Severe hallucination** | Q1 (Paris in US), Q8 (Krasnarol), Q10 (Rentalist) — every factual question produces wrong answers |
| **Math inability** | Q4: 375 vs polynomial rambling — neither close to 555 |
| **Instruction blindness** | Q5: ignored "one sentence"; Q11: ignored "haiku" |
| **Small model ceiling** | 73.5M params is far below the scale needed for factual grounding |

### sft-specific

| Weakness | Examples |
|---|---|
| **Longer rambling** | Q1 continues after stating Paris with geographic hallucination; Q5 (6.2s) |
| **Confidence in wrong facts** | More likely to assert incorrect claims with confidence |

### d6-specific

| Weakness | Examples |
|---|---|
| **Tautology** | Q1: "Paris is the capital of France, the capital of France" |
| **Non-answers** | Q4: polynomial rambling instead of a number |
| **Hallucinated entities** | Q8: invents "Krasnarol" — not a real location |

---

## 7. Recommendations

### Short-term (same architecture)

1. **Improve instruction tuning** — Both models consistently ignore format constraints ("one sentence," "haiku"). Adding supervised examples that reward structural compliance would help.
2. **Factual grounding via retrieval** — At 73.5M params, these models cannot memorize facts reliably. A retrieval-augmented generation (RAG) pipeline or a tool-use loop (e.g., Wikipedia lookup) would dramatically improve factual QA.
3. **Calibration training** — sft confidently asserts wrong facts; training with a calibration objective (or rejection sampling) could reduce hallucination severity.

### Medium-term (scaling)

4. **Larger model** — 73.5M params is the dominant limitation. Scaling to 350M–1B params with more training data is the most direct path to reduced hallucination.
5. **More diverse training data** — The code/tool-use strength suggests the training mix includes high-quality code data. Increasing the proportion of factual, instructional, and formatted-output data would balance the weaknesses.
6. **Evaluation-driven development** — Build this 12-question benchmark into the training loop. Track category-level scores to catch regressions early.

### Long-term

7. **RLHF / DPO alignment** — Once the base model is larger, alignment techniques could teach the model to decline answering when uncertain, following the pattern of modern instruction-tuned LLMs.

---

## Appendix: Raw Results

See `results.json` in this directory for the complete structured output data.
