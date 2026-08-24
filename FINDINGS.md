# ConCor-1 Evaluation — Failure Mode Testing

**Author:** Akash Thorali
**Model:** [ConCor-1](https://github.com/RAIVNLab/ConCor) (Zhang, Gao, Zettlemoyer, Krishna 2026)
**Test notebook:** [INSERT LINK TO NOTEBOOK]

**Goal**: Identify failures in ConCor-1's correspondence predictions. 

## Test Case A — Same-category disambiguation

Tests whether the model can differentiate multiple instances of the same object using caption context (e.g two cats) rather than collapsing them into one or confusing which is which. 

### A1 — Disambiguate by position

Status: Partial failure (text segmentation)

Text: "The cat on the left is curled up asleep. The cat on the right is also lying down. Between them are two black remote controls."

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 8 | 0.999 | "The cat" | (145, 229) | 52,758 |
| 2 | 0.996 | "The cat" | (463, 185) | 60,183 |
| 149 | 0.999 | "two black remote controls" | (104, 93) | 4,260 |
| 320 | 0.677 | "two black remote controls" | (349, 124) | 2,003 |

Correspondence pairing and image masks correctly separated the two cats, but the text segmentation head truncated both spans to "The cat," dropping the "on the left" and "on the right", the only two words that differentiate the two mentions of the word. As such, both correspondences of the word are indistinguishable from their text phrase despite being spatially correct. Additionally, the plural "two black remote controls" split into two per-instance correspondences at different locations, yet the second scored notably lower, indicating hedged confidence when splitting a plural referent. 

