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
- [ ] TASK-008: Full transformer block — multi-head attention, feed-forward
  layer, layernorm, residual connections, assembled and unit-tested as one
  block with a known input/output shape.
  Depends on: TASK-005, TASK-006, TASK-007

## Model assembly
- [ ] TASK-009: Decoder-only (GPT-style) model — stack N transformer blocks
  from TASK-008, add the output projection back to vocabulary logits over
  the tokeniser's vocabulary (TASK-001).
  Depends on: TASK-008

## Training loop
- [ ] TASK-010: Batching — turn a tokenised corpus (TASK-001/002 output)
  into fixed-length training batches (input/target pairs shifted by one
  token) for next-token prediction. Stream/index batches from the
  disk-backed token store rather than loading the full tokenised corpus
  into RAM at once.
  Depends on: TASK-002
- [ ] TASK-011: Cross-entropy loss + optimizer (SGD first, then AdamW) built
  on the autodiff engine.
  Depends on: TASK-004
- [ ] TASK-012: Training loop — wires TASK-009/010/011 together: forward
  pass, loss, backward pass, optimizer step, checkpointing (save/load model
  weights), basic logging of loss over time. Checkpoints and optimizer
  state (e.g. Adam's moment estimates, which double memory vs. the raw
  weights) are candidates for disk-backed storage once model size makes
  them non-trivial.
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
