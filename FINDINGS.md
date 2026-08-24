# ConCor-1 Evaluation — Failure Mode Testing

**Author:** Akash Thorali
**Model:** [ConCor-1](https://github.com/RAIVNLab/ConCor) (Zhang, Gao, Zettlemoyer, Krishna 2026)
**Test notebook:** [INSERT LINK TO NOTEBOOK]

**Goal**: Identify failures in ConCor-1's correspondence predictions. 

## Test Case A — Same-category disambiguation

Tests whether the model can differentiate multiple instances of the same object using caption context (e.g two cats) rather than collapsing them into one or confusing which is which. 

### A1 — Disambiguate by position

**Status: Partial failure (text segmentation)**

Text: "The cat on the left is curled up asleep. The cat on the right is also lying down. Between them are two black remote controls."

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 8 | 0.999 | "The cat" | (145, 229) | 52,758 |
| 2 | 0.996 | "The cat" | (463, 185) | 60,183 |
| 149 | 0.999 | "two black remote controls" | (104, 93) | 4,260 |
| 320 | 0.677 | "two black remote controls" | (349, 124) | 2,003 |

**Success**: Correspondence pairing and image masks correctly seperated the two cats.

**Failure**: The text segmentation head truncated both spans to "The cat," dropping the "on the left" and "on the right", the only two words that differentiate the two mentions of the word. Thus, both correspondences of the word are indistinguishable from their text phrase despite being spatially correct. Additionally, the plural "two black remote controls" split into two per-instance correspondences at different locations, yet the second scored notably lower, indicating a hedged confience on a plural split. 

[INSERT IMAGE]

### A2 — Pronoun coreference chain

**Status: Mixed — coreference correct, duplicate-mask failure found**

Text: "A cat lies on the couch. It is grey and white. Next to it is another
cat, and they are both resting near two remote controls."

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 8 | 0.998 | "A cat" | (145, 229) | 52,944 |
| 2 | 0.994 | "another cat" | (463, 185) | 60,149 |
| 149 | 0.980 | "two remote controls" | (104, 92) | 4,250 |
| 0 | 0.707 | "cat" | (327, 265) | **183,878** |
| 310 | 0.281 | "two remote controls" | (348, 124) | 2,003 |

**Success:** Correctly resolved to the same cat as "A cat" and "another cat" was properly identified as the second, distinct cat.

**Failure:** Low-confidence duplicate correspondence for the bare word "cat" since a truncated re-mention of the already-captured "A cat" is not a new entity. Its mask is degenerate: 183,878px, roughly 3x larger than either legitimate cat mask, covering ~60% of the frame. Text segmentation produced a redundant span, image segmentation produced an invalid oversized mask, and the presence head only partially down-weighted it. 

## Test Case B — Hallucinated / absent entities

Tests whether the presence head tracks what's actually in the image or if it hallucinates matches for mentioned but absent objects or overgrounds on visible but unmentioned ones. 

### B1 — Entity mentioned but not present ("a dog")

**Status: PASS**

Text: "A dog sits next to two cats on a couch."

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 2 | 0.999 | "two cats" | (462, 186) | 58,714 |
| 8 | 0.995 | "two cats" | (145, 229) | 51,168 |
| 0 | 0.874 | "a couch" | (327, 266) | 183,640 |

**Success:** No correspondence was produced for "a dog" despite being named explicitly, and no mask was hallucinated for an absent entity. "two cats" split into two confident, correctly-positioned correspondences, this time with no hedging as opposed to A1.

**Failure**: None for this case.

[INSERT IMAGE]

### B2 — Object present but not mentioned (cats, when only remotes are named)

**Status: PASS**

Text: "Two remote controls rest on a couch."

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 149 | 0.996 | "Two remote controls" | (104, 93) | 4,277 |
| 320 | 0.967 | "Two remote controls" | (349, 125) | 2,036 |
| 0 | 0.261 | "a couch" | (327, 264) | 186,747 |

**Success**: Despite not being mentioned, the clearly visible cats did not generate any correspondence. No over-grounding on a salient but unmentioned object.

**Failure**: None for this case.

**Cross-case note:** the "a couch" correspondence has a near-identical mask
in both B1 and B2 (area ~184–187k, same centroid), but presence score swings
from 0.874 to 0.261 across the two prompts. Same object, same mask, same
phrase — the only variable is unrelated surrounding sentence content. This
instability wasn't observed for foreground objects (cats, remotes) in any
test — only for the couch.

[INSERT IMAGE]

## Test Case C — Negation

Tests whether the model can use a negated clause to disambiguate between two similar candidates (not discussed in paper). 

### C1 — "The cat that is not sleeping is looking toward the camera."

**Status: FAIL — clearest failure in this evaluation**

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 2 | 0.412 | "The cat" | (463, 186) | 59,081 |
| 8 | 0.231 | "The cat" | (145, 229) | 51,628 |

**Success**: Masks correctly landed on the two distinct, correctly positioned cats.

**Failure**: Both bridge tokens produced the identical truncated span "The cat" and the negation clause "that is not sleeping" was dropped from both. This failure is more severe than the one in A1 since the missing words here determine which cat is even being referred to. Low confidence splits across both (0.412, 0.231), significantly below any legitimate single-answer correspondence elsewhere in this evaluation. Reads as the model recognizing ambiguity but having no mechanism to resolve it using the negation.

[INSERT IMAGE]

## Test Case D — Compositional/relational language

Tests whether the model can form a spatial relation between two objects to correctly resolve which one is being referred to. 

### D1 — "The remote control closest to the sleeping cat's paw."

**Status: PASS — strongest positive result in this evaluation**

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 320 | 0.879 | "The remote control" | (349, 125) | 2,038 |
| 2 | 0.702 | "the sleeping cat" | (462, 186) | 58,484 |
| 149 | 0.155 | "The remote control" | (105, 93) | 4,283 |

**Success:** Correctly produced a separate correspondence for the contextual entity ("the sleeping cat") alongside the target ("the remote control"). The high-confidence remote candidate (0.879, x=349) sits ~3x closer to "the sleeping cat" (x=462) than the low-confidence candidate (0.155, x=105), indicating the spatial region was formed rather than resolved arbitrarily. 

**Failure**: None for this case.

## Test Case E — Large category-list vocabulary (LVIS-style)

Tests single-pass precision over a large candidate list where most categories are absent.

### E1 — 20-category list, only 3 present

**Status: PASS**

Text: `cat . remote control . couch . dog . person . television . blanket . pillow . laptop . book . lamp . plant . window . curtain . rug . mirror . vase . clock . bird . fireplace`

| Bridge | Score | Text phrase | Centroid | Area |
|---|---|---|---|---|
| 2 | 0.993 | "cat" | (464, 185) | 61,258 |
| 149 | 0.980 | "remote control" | (105, 93) | 4,390 |
| 8 | 0.954 | "cat" | (146, 230) | 54,084 |
| 0 | 0.926 | "couch" | (326, 264) | 189,539 |
| 320 | 0.378 | "remote control" | (349, 125) | 2,076 |
| 280 | 0.170 | "blanket", "pillow" (merged) | (326, 479) | 625 |

**Success:** of 20 candidate categories, only 3 are present, and all 17 absent categories were correctly rejected, with no hallucinations.

**Failure:** a low-confidence (0.170) correspondence merging two different absent categories, "blanket" and "pillow," into one small (625px) mask at the bottom edge of the frame. Likely a low-confidence texture guess.

## Summary and takeaways

### What held up

- **Absence calibration:** reliably avoided hallucinating masks for mentioned-but-absent entities (B1, E1), and avoided over-grounding visible but not mentioned entities (B2).
- **Basic coreference:** pronoun resolution and repeated mention merging worked correctly.
- **Instance-level disambiguation:** every multi-cat/multi-remote test produced spatially distinct, correctly positioned masks per instance.
- **Compositional spatial reasoning (D1):** correctly resolved a two-object spatial relation not explicitly described in the paper's task formulation.

### What broke down

- **Text segmentation under-captures disambiguating modifiers.** "on the left"/"on the right" (A1) and the full negation clause "that is not sleeping" (C1) were both dropped from their spans, even where the underlying object detection was correct. Negation (C1) escalates this issue. 
- **Presence-head instability for background/supporting-surface objects.** The couch swung from 0.874 to 0.261 across two prompts with an essentially identical mask, not observed for foreground objects. 
- **Occasional duplicate/degenerate correspondences (A2).** A redundant, oversized mask for a re-mention of an already-grounded entity, only partially discounted by the presence of the head.
- **Plural-instance confidence hedging.** Splitting a plural referent into per-instance correspondences sometimes hedges the second instance's confidence (A1, D1) and sometimes doesn't (B1). 

