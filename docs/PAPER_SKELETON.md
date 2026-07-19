# Paper Skeleton — Working title options:
# (a) "Do Conversation Graphs Survive Fine-Tuned Encoders? A Controlled
#      Attribution Study of Multimodal Emotion Recognition in Conversation"
# (b) "RAPPORT: When Does Relationship Memory Matter? Contextual Encoders,
#      Graph Machinery, and an Attribution Methodology for Multimodal ERC"
# Choose (a) if IEMOCAP is also null; (b) if IEMOCAP rescues relational memory.
# Target: IEEE TAFFC or Information Fusion if (a); Neurocomputing viable if (b).
# All venues: no APC on subscription route.

## Abstract (two variants, finalize after N5-B)

VARIANT-NULL (IEMOCAP negative):
Graph-based context modeling is the dominant architectural paradigm in
multimodal emotion recognition in conversation (ERC), yet its gains are
typically measured over frozen or weakly adapted utterance encoders. We
present a controlled attribution study asking whether conversation-graph
machinery — speaker-state recurrence, relationship (edge) memory, and
shift-aware auxiliary objectives — retains measurable value once the text
encoder is fine-tuned. Using a residual "do-no-harm" attribution design in
which every architectural component provably initializes as an identity on
a frozen fine-tuned foundation, we find across two benchmarks (MELD,
IEMOCAP) and up to seven seeds that: (i) on frozen features, graph context
modeling yields consistent gains, reproducing the literature; (ii) after
LoRA fine-tuning of the text encoder — even WITHOUT conversational context
in its input window — these gains vanish or reverse; (iii) contextual
input windows further subsume what remains. Our results suggest that a
substantial fraction of reported graph-architecture gains in ERC
compensate for representation deficiencies rather than model social
structure, and we release our attribution methodology as a reusable
protocol. [~180 words; add final numbers]

VARIANT-CONDITIONAL (IEMOCAP positive):
[Same setup, then:] ...we find that relationship memory contributes no
measurable gain on short multi-party scenes (MELD) but yields consistent
improvements on long dyadic interactions (IEMOCAP), localizing the value
of relational modeling to conversations long enough to develop
relationship trajectories beyond the encoder's context window. [RAPPORT
becomes the named model; the null MELD result becomes the boundary
condition that motivates it.]

## 1. Introduction
- ERC matters for digital services (recycle TSUE-paper motivation, brief).
- The field's trajectory: isolated utterances -> recurrent context ->
  graph context (DialogueGCN, MMGCN, COGMEN, SocialArcNet [own prior
  work, cited at published 0.62]).
- The unexamined assumption: graph gains are measured over frozen/weak
  encoders. Question: do they survive encoder fine-tuning?
- Contributions (order depends on branch):
  C1. Residual attribution methodology: every component initializes as a
      provable identity on a frozen fine-tuned foundation; deltas are
      attributable by construction. (Methodological contribution.)
  C2. The fine-tuning boundary finding: graph machinery's gains on frozen
      features vs. their disappearance under LoRA-tuned encoders,
      with the k-sweep showing context windows subsume the remainder.
  C3. Relational (pair-state) memory + shift-aware objective: a fair,
      pre-registered trial [contribution if IEMOCAP positive; analyzed
      negative result with mechanisms if not].
  C4. Full pre-registered, seed-replicated, publicly versioned protocol
      (gates, locked recipes, per-decision audit trail in-repo).

## 2. Related Work
- 2.1 Graph/contextual ERC architectures (the machinery under test).
- 2.2 Contextual text encoding in ERC (EmoBERTa, CoMPM lineage) — the
  regime change this paper leverages.
- 2.3 Parameter-efficient adaptation (LoRA) in affective computing.
- 2.4 Negative results & attribution methodology in deep learning
  (ablation-validity critiques) — positions C1.

## 3. Attribution Methodology (the paper's methodological core)
- 3.1 Foundation: LoRA-tuned RoBERTa, context window k, masked-mean
  pooling, plain CE + post-hoc logit adjustment (report the trained-LA
  negative finding here as an implementation note: trained logit
  adjustment ranked rare classes last where plain CE + post-hoc did not).
- 3.2 Residual design: z = z_text + W_out g(...), W_out and cross-modal
  blocks zero-initialized; equality-at-init property test parametrized
  over all component-flag combinations. Why this defeats the
  "new-component-disturbed-training" confound.
- 3.3 Components under test: speaker-state GNN; relational pair-state
  memory (spec v1.1); shift-aware auxiliary loss; temporal attention.
- 3.4 Protocol: pre-registered gates, locked recipe (RECIPE.md), fixed
  seed sets, write-on-produce logging, public repo with full history.

## 4. MELD Experiments
- 4.1 Setup, class statistics, splits.
- 4.2 Frozen-feature era: GNN vs context-free probes (+ gain) — the
  literature's regime, reproduced. [State feature-set caveat OR include
  the bridging experiment if run.]
- 4.3 Fine-tuned era: the n=7 endpoint results (k=0 and k=8 both null/
  negative), full ablation matrix, per-class analysis.
- 4.4 The k-sweep figure (centerpiece): interior points de-emphasized,
  endpoints with n=7/n=5 error bars; paired per-seed gains.
- 4.5 Mechanism analysis: overfitting gap growth with component count;
  fusion weight norms (what fusion learned to suppress).

## 5. IEMOCAP: The Fair Trial for Relationship Memory  [PENDING N5-B]
- Pre-registered hypothesis (quote it verbatim from PHASE_N5B.md, with
  commit hash predating the data download — this is a genuine
  pre-registration and should be presented as one).
- Setup: label protocol, Session5 test / Session4 val, dyadic N=2
  degeneracy note.
- Results: [branch point]. Edge-state trajectory figure over sample
  sessions (qualitative) if positive.

## 6. Analysis of Negative Results (or Boundary Conditions)
- Relational memory & shift objective on MELD: why they didn't help —
  scene length, premise statistics from SHIFT_LABEL_STATS.md, the
  redundancy probe. What WOULD falsify the null (longer dialogues,
  relationship-annotated corpora).
- Threats to validity: n<=7 seeds, single foundation model family,
  MELD/IEMOCAP domain limits, frozen A/V features (probe ceilings 0.41 /
  0.33 documented), the frozen-vs-tuned feature-set caveat in 4.2.

## 7. Conclusion
- The boundary claim; the methodology as reusable protocol; implications
  for how ERC architecture papers should baseline going forward.

## Compliance checklist (from license terms, do not lose):
- Cite Busso et al. 2008 (IEMOCAP) + the usage-guidance paper from the
  credentials email (ieeexplore 10096808).
- SAIL acknowledgment per release form clause 5.
- Clause 6: share planned evaluations/results with SAIL before public
  reporting — send them a courtesy summary at submission time.
- TUIT grant acknowledgment if institutional policy expects it.
- Data availability statement: MELD public; IEMOCAP by license; code +
  full experimental history at github.com/multi-modal-rtm/rapport.
