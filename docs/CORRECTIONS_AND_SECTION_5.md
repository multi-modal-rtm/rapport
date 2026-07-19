# Draft Corrections (from DRAFT_NUMBER_AUDIT.md) + Section 5 Draft
*(Apply the replacements to the committed drafts verbatim, then re-run the audit — target: all green.)*

## Part 1 — Corrections to Sections 3-4

### C1. Section 4.5 — REWRITE the mechanism paragraph (substantive; the suppression claim was contradicted by the record)

REPLACE the sentence beginning "Second, what fusion learns: the trained fusion projector's weight norms..." WITH:

> Second, what fusion learns: the trained fusion projector's weight norms distribute nearly evenly across the textual, acoustic, and visual blocks (34%, 32%, and 33% respectively), so the stack does not learn to gate out the weaker modalities — it integrates them with essentially uniform weight and gains nothing by doing so. This rules out one tempting explanation of the null (that the model defensively suppresses noisy inputs and therefore never accesses their signal) and leaves the starker one: the acoustic and visual features are fully present in the fused representation, and the rare-class signal the balanced probes locate in them (Section 4.1) does not survive translation into utterance-level predictions once the textual foundation is strong.

### C2. Section 4.3 — comparator mislabel (the −0.0048 is vs. the fusion baseline, not the anchor)

REPLACE: "the full stack's paired gain over the fine-tuned anchor is **−0.0048** at the locked context width"
WITH: "the full stack's paired gain over its own fusion baseline is **−0.0048** at the locked context width, and over the fine-tuned text anchor **−0.0060**"

(The 63%-shrinkage comparison in 4.5 remains correct as written — both its endpoints are full-vs-baseline figures.)

### C3. Section 4.2 — remove the unfulfillable robustness footnote

DELETE: "[CHECK: add the weighted-F1-selection robustness re-scoring as a footnote if run.]"
The audit confirms per-epoch checkpoints were not retained, so re-selection under a different rule is not possible without a new campaign, which the closed-record policy forbids. The internal-consistency argument in the same paragraph (anchor and stack share the selection rule; the fine-tuned regime uses the identical rule and still yields nulls) carries the point on its own and stays.

### C4. Rounding/count fixes

- "1,433 multi-party dialogues" -> "1,432 multi-party dialogues"
- "48.2% of the test set" -> "48.1% of the test set"

### C5. External-baseline citations (SocialArcNet, DialogueRNN, MMGCN, EmoShiftNet)

Everywhere these numbers appear (3.1, 4.3), phrase them explicitly as *published* results with citations — e.g., "exceeding the *published* performance of several complete multimodal graph systems [cites]" — and add one sentence to Section 6's threats-to-validity:

> Literature comparisons in this paper cite published numbers and are not independently reproduced; in particular, the original implementation of our own prior system was lost to an infrastructure failure (documented in the repository), and our early re-implementation attempts are not faithful reproductions.

This converts the repo's honest record of the 0.3765 episode from a liability into a disclosed limitation, and prevents any reviewer discovering it in the public history and wondering why the paper is silent.

---

## Part 2 — Section 5 Draft

# Section 5 — IEMOCAP: A Pre-Registered Trial for Relationship Memory

The MELD results admit one principled defense of relational modeling: MELD's dialogues are short multi-party scenes, averaging roughly nine utterances, in which there is scarcely any relationship *trajectory* to remember — perhaps relationship memory fails there not because the mechanism is empty but because the corpus cannot exercise it. IEMOCAP [Busso et al., 2008] is the natural test of this defense: long dyadic sessions of emotionally volatile scripted and improvised interaction between two actors, precisely the setting in which a per-pair recurrent memory should accumulate signal that an eight-utterance context window cannot hold. We committed to this test before we possessed the data: the hypothesis, its reading rule, and the paper-branching decision it governs were written into the public repository (commit 04d9267) prior to the corpus download, and we report them here unaltered.

**Pre-registered hypothesis** (quoted verbatim from the repository): [INSERT the exact hypothesis text from docs/PHASE_N5B.md at commit 04d9267 — agent task]. In summary form: the relational component's paired contribution on IEMOCAP was predicted to exceed its contribution on MELD, driven by dialogue length. The pre-registered decision rule: if the relational component's isolated paired gain (full stack minus the same stack without relational memory) is positive and exceeds one standard deviation, the paper's framing follows the conditional-value branch; otherwise it follows the boundary branch reported here.

**Setup.** We use the six-class categorical protocol {angry, happy, excited, sad, neutral, frustrated}, retaining 7,380 of 10,039 utterances — chosen in advance of any training for three stated reasons: maximal test power for the paired comparison (1,622 test utterances versus 1,241 under the four-class alternative), comparability with the multimodal ERC literature, and retention of *frustrated*, the corpus's most interaction-driven label and therefore the fairest possible class for the relational hypothesis. Sessions 1–3 train, Session 4 validates, Session 5 tests; this fixed split is stated rather than tuned, and we caution that published IEMOCAP numbers vary substantially with split convention. Preprocessing cuts per-utterance audio and speaker-cropped video from the session recordings using the corpus's own annotation timestamps; the fine-tuned textual foundation is retrained on IEMOCAP under the locked recipe, attaining 0.579 ± 0.016 test weighted F1 with all six classes nonzero in every seed. One recipe note: the inference-time logit-adjustment temperature selected on MELD produced no qualifying candidate on IEMOCAP's validation distribution under the locked selection rule, so raw logits are used throughout this section for every configuration identically.

**Result.** Under the identical residual apparatus — equality-at-init verified against the IEMOCAP foundation before launch, three seeds, the same configurations — the relational component's isolated paired contribution is **−0.0334 ± 0.0284** on IEMOCAP, against −0.0022 ± 0.0086 on MELD. The hypothesis is refuted, and refuted in the diagnostic direction: longer dialogues did not give relationship memory room to help; they gave it room to hurt. The class-level evidence is congruent — on *frustrated*, the pre-registered best case, the full stack scores 0.561 where the stack without relational memory scores 0.610. The decision rule fires the boundary branch mechanically, and this paper's framing is its consequence rather than its premise.

**A qualitative window.** Figure [EDGE-NORM] traces the relational edge-state norm across three complete Session 5 dialogues. The trajectories are structured rather than degenerate — the memory is demonstrably accumulating something over the interaction [CHECK against the generated figure: describe the observed pattern honestly] — which anticipates the dissociation developed in Section 6: the relational states carry genuine conversational-dynamics information (they *improve* the auxiliary shift-detection task on both corpora) that the emotion readout, resting on a fine-tuned foundation, cannot convert into classification gains.

**What this section establishes.** The conditional defense of relational modeling — "it would work where relationships have time to develop" — was given its best pre-registered shot: the corpus chosen for it, the protocol chosen to favor it, the class retained for it, and the reading rule fixed before the data existed. The verdict extends the MELD boundary claim rather than qualifying it: on fine-tuned foundations, conversation-graph machinery, including explicit relationship memory, does not add measurable value on either short multi-party scenes or long dyadic trajectories.
