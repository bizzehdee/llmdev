# TASK.md

Standing constraint on every task below (see PLAN.md "Memory constraint"):
use disk as scratch space and minimise RAM usage for anything that scales
with corpus/model/batch size. Prefer disk-backed/streamed structures over
large heap allocations — this machine has OOM-killed itself on this project
before. Flagged explicitly on the tasks it matters most for below.

- [x] TASK-001: Byte-level BPE tokeniser (train, encode, decode, save/load)
- [x] TASK-002: Directory input + disk-backed scratch for large corpora

## Tensor + autodiff engine
Required by: TASK-005, TASK-006, TASK-007, TASK-008, TASK-009

- [x] TASK-003: `Tensor` type — N-dimensional float array (shape, strides,
  indexing) with elementwise ops (add, subtract, multiply, divide) and
  matmul, transpose, sum/mean-along-axis, broadcasting. Backing storage
  should be swappable between an in-heap array (small tensors) and a
  disk-backed store (large ones) so later tasks aren't locked into
  unbounded heap growth — see `src/Tokeniser/MappedInt32Array.cs` for the
  established pattern. Done: `src/Tensor/`, backed by the shared
  `Common.MappedArray<T>` (extracted from the tokeniser's
  `MappedInt32Array` as part of this task, now used by both). Ops always
  produce heap-backed results for now — routing large op results to disk
  is left to whichever later task first needs it.
  Depends on: none
- [x] TASK-004: Reverse-mode autodiff — computation graph recording ops as
  they run, `Backward()` to propagate gradients, gradient accumulation.
  Covers the ops from TASK-003 plus softmax, exp/log, and the activation
  functions transformer blocks need (GELU or ReLU). Done: `src/Tensor/Variable*.cs`
  — chose ReLU over GELU (PLAN.md's "either" framing) to keep the backward
  math simple; softmax has no max-subtraction stability trick yet (no
  max-along-axis reduction in Tensor), noted as a revisit candidate if large
  logits cause overflow. Tested via finite-difference gradient checking
  (`tests/Tensor.Tests/VariableTests.cs`) rather than hand-derived expected
  values, plus explicit tests for gradient accumulation (a variable used
  twice, and a diamond dependency).
  Depends on: TASK-003

## Embeddings
Required by: TASK-008

- [x] TASK-005: Token embedding table — learned lookup from token id to a
  dense vector, backed by `Tensor`, trainable via the autodiff engine.
  Done: `src/Model/TokenEmbedding.cs`, on a new differentiable
  `Tensor.GatherRows`/`ScatterAddRows` pair (`Variable.GatherRows`) added
  to the tensor engine for this — row lookup with gradient accumulation on
  repeated indices, which is what makes a token appearing twice in one
  sequence get the sum of both occurrences' gradients rather than losing
  one. Weights init as small Gaussian noise (mean 0, std 0.02, GPT-2's
  scale) via a hand-rolled Box-Muller transform, not zeros — identical
  rows would otherwise get identical gradients forever.
  Depends on: TASK-004
- [x] TASK-006: Positional encoding — learned positional embeddings added to
  token embeddings (start with learned rather than sinusoidal/RoPE: fewer
  moving parts for a first pass). Done: `src/Model/PositionalEmbedding.cs`
  — same shape as `TokenEmbedding` (a `[maxSequenceLength, embeddingDim]`
  trainable table, looked up via the same `GatherRows` op), sized to a max
  sequence length rather than generalising to arbitrary lengths like
  sinusoidal/RoPE would. Shared the Gaussian weight-init logic with
  `TokenEmbedding` via a new `GaussianInit` helper instead of duplicating
  it. Composition with `TokenEmbedding` (plain elementwise `Add`, both
  produce `[sequenceLength, embeddingDim]`) is covered by tests but not
  wrapped in its own class yet - left for whichever later task first
  assembles a full embedding layer.
  Depends on: TASK-004

## Attention + transformer block
Required by: TASK-008

- [x] TASK-007: Scaled dot-product attention + multi-head attention.
  Done: `src/Model/ScaledDotProductAttention.cs` (parameter-free,
  batched over any leading dims so it takes numHeads as a batch
  dimension) and `src/Model/MultiHeadAttention.cs` (owns the Q/K/V/O
  projection weights, splits into heads via `Reshape`+`Transpose`,
  concatenates back). Includes causal masking (additive -infinity mask
  before softmax) since the model this feeds is decoder-only/autoregressive
  — without it, training would let a position "see" the token it's meant
  to predict. No projection bias terms (kept to what's needed; feed-forward
  in TASK-008 needs bias, attention doesn't strictly). Added
  `Variable.Reshape` (mirrors `Tensor.Reshape`) to the autodiff engine to
  support head-splitting. Tested via finite-difference gradient checks
  (including against persistent weights already embedded in a module, a
  different shape of check than TASK-004's) and explicit causal-masking
  correctness tests (changing a future position's value must not change an
  earlier position's output; contrasted against non-causal where it does).
  Depends on: TASK-004
- [x] TASK-008: Full transformer block — multi-head attention, feed-forward
  layer, layernorm, residual connections, assembled and unit-tested as one
  block with a known input/output shape. Done:
  `src/Model/{LayerNorm,FeedForward,TransformerBlock}.cs`, GPT-2-style
  pre-norm layout (normalise -> attend -> +residual -> normalise ->
  feed-forward -> +residual). FeedForward expands to `4*embeddingDim`
  hidden units by default (GPT-2's convention), ReLU in between (per
  TASK-004's choice over GELU). Added `Tensor.Sqrt`/`Variable.Sqrt` to the
  tensor engine for layernorm's variance normalisation. Tested end to end:
  causal masking survives being wrapped in a full block, gradient checks
  against a representative parameter in every sub-layer (not just one),
  and a per-position-independence check for both layernorm and
  feed-forward (no cross-position mixing outside of attention).
  Depends on: TASK-005, TASK-006, TASK-007

## Model assembly
- [x] TASK-009: Decoder-only (GPT-style) model — stack N transformer blocks
  from TASK-008, add the output projection back to vocabulary logits over
  the tokeniser's vocabulary (TASK-001). Done: `src/Model/GptModel.cs` —
  token+positional embeddings, N causal `TransformerBlock`s, a final
  layernorm (GPT-2's "ln_f"), then an output projection that reuses
  (rather than duplicates) `TokenEmbedding.Weight`, transposed - weight
  tying, same as GPT-2 itself. Tested end to end: causal masking survives
  the full model (a later token can't change an earlier position's
  logits), weight tying verified by checking the output projection's
  gradient reaches embedding rows that were never looked up as input
  tokens (proof it's the same matrix, not just equal by coincidence), and
  finite-difference gradient checks against parameters spanning every
  stage (token/positional embeddings, a block's attention weight, the
  final norm).
  Depends on: TASK-008

## Training loop
- [x] TASK-010: Batching — turn a tokenised corpus (TASK-001/002 output)
  into fixed-length training batches (input/target pairs shifted by one
  token) for next-token prediction. Stream/index batches from the
  disk-backed token store rather than loading the full tokenised corpus
  into RAM at once. Done: new `src/Training/` project —
  `TokenCorpus` (a token-id stream in a disk-backed `MappedArray<int>`,
  reusing `Common`'s pattern) and `BatchSampler` (draws fixed-length
  input/target windows from it; batch memory is O(batchSize*contextLength),
  never O(corpus length)). `GptModel` (TASK-009) has no batch dimension of
  its own, so a "batch" here is just a set of examples for TASK-012's
  training loop to accumulate gradients over, not a single batched tensor
  op. Known gap: getting token ids into `TokenCorpus` in the first place
  still goes through `BpeTokeniser.Encode`, which (unlike `Train`) uses the
  simple merge-scan approach tuned for short/moderate text, not efficient
  bulk encoding of a large corpus - noted here rather than solved, since
  it's outside batching itself; revisit if an actual training run proves
  it's too slow in practice.
  Depends on: TASK-002
- [x] TASK-011: Cross-entropy loss + optimizer (SGD first, then AdamW) built
  on the autodiff engine. Done: `src/Training/{CrossEntropyLoss,SgdOptimizer,AdamWOptimizer}.cs`.
  Loss computes log-softmax directly (x - log(sum(exp(x)))) rather than
  Softmax().Log(), to avoid round-tripping through a softmax value that
  could already have underflowed to 0 - same no-max-subtraction caveat as
  `Variable.Softmax` still applies to the sum(exp(x)) term itself, though.
  Needed two new tensor primitives: `Tensor.GatherColumns`/`ScatterAddColumns`
  (pick one column per row - "the log-probability assigned to the correct
  next token" - and its gradient-scattering inverse), and
  `Tensor.SubtractInPlace`, the one deliberate mutation primitive in the
  whole tensor engine (every other op returns a new tensor): needed
  because a `Variable.Value` can't be reassigned once an optimizer holds
  that Variable by reference. AdamW's moment-estimate state is plain heap
  tensors for now (~2x parameter count); TASK-012 decides whether that
  needs disk-backing once model size makes it non-trivial (already flagged
  there in PLAN.md). Tested via known-exact-value checks (uniform logits
  give loss = log(vocabSize) exactly), gradient-direction sanity, finite-
  difference gradient checks, a hand-computed single AdamW step, and two
  end-to-end convergence tests (minimising a toy quadratic, and driving
  down cross-entropy loss against fixed targets over many steps).
  Depends on: TASK-004
- [x] TASK-012: Training loop — wires TASK-009/010/011 together: forward
  pass, loss, backward pass, optimizer step, checkpointing (save/load model
  weights), basic logging of loss over time. Checkpoints and optimizer
  state (e.g. Adam's moment estimates, which double memory vs. the raw
  weights) are candidates for disk-backed storage once model size makes
  them non-trivial. Done: `src/Training/{Trainer,ModelCheckpoint}.cs`, plus
  a new `GptModel.Parameters()` (and matching `Parameters()` on every
  sub-layer, composed bottom-up) so an optimizer/checkpoint can enumerate
  every trainable Variable - explicitly excludes the weight-tied output
  projection to avoid double-counting it. `Trainer.Step` accumulates
  gradients over a batch by scaling each example's loss by 1/batchSize
  before its own `Backward()` call (rather than summing raw gradients and
  scaling after), keeping the effective step size independent of batch
  size using only existing autodiff ops. Logging is a plain
  `Action<int,float>? onStep` callback (mirrors `BpeTokeniser.Train`'s
  `onMerge`), not hardcoded console output. Checkpointing is binary
  (shape + values per parameter, hyperparameters in the header so `Load`
  reconstructs the exact architecture before overwriting values in place)
  - deferred rather than solved: still plain heap allocation, no
  disk-backing yet for checkpoints or AdamW's moment estimates, since
  nothing in this project has been large enough yet to need it; revisit
  if that changes. Tested via checkpoint round-trip (identical parameter
  values and identical forward-pass output before/after save+load,
  corrupted-file rejection) and a genuine end-to-end training run: loss
  measurably drops on a small repetitive corpus over 100 steps, not just
  a unit-level check of each piece in isolation.
  Depends on: TASK-009, TASK-010, TASK-011

## Generation
- [ ] TASK-013: Sampling — greedy, temperature, and top-k/top-p decoding
  from a trained (or in-progress) model, using the tokeniser to decode
  output token ids back to text.
  Depends on: TASK-012

## Notes
- Tasks are scoped for hand-rolled, no-library implementation per PLAN.md.
- Work through tasks one at a time in order; ambiguities get clarified
  before implementation starts on that task, not up front for all of them.
