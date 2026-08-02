# TASK.md

- [x] TASK-001: Byte-level BPE tokeniser (train, encode, decode, save/load)
- [x] TASK-002: Directory input + disk-backed scratch for large corpora

## Tensor + autodiff engine
Required by: TASK-005, TASK-006, TASK-007, TASK-008, TASK-009

- [ ] TASK-003: `Tensor` type — N-dimensional float array (shape, strides,
  indexing) with elementwise ops (add, subtract, multiply, divide) and
  matmul, transpose, sum/mean-along-axis, broadcasting.
  Depends on: none
- [ ] TASK-004: Reverse-mode autodiff — computation graph recording ops as
  they run, `Backward()` to propagate gradients, gradient accumulation.
  Covers the ops from TASK-003 plus softmax, exp/log, and the activation
  functions transformer blocks need (GELU or ReLU).
  Depends on: TASK-003

## Embeddings
Required by: TASK-008

- [ ] TASK-005: Token embedding table — learned lookup from token id to a
  dense vector, backed by `Tensor`, trainable via the autodiff engine.
  Depends on: TASK-004
- [ ] TASK-006: Positional encoding — learned positional embeddings added to
  token embeddings (start with learned rather than sinusoidal/RoPE: fewer
  moving parts for a first pass).
  Depends on: TASK-004

## Attention + transformer block
Required by: TASK-008

- [ ] TASK-007: Scaled dot-product attention + multi-head attention.
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
  token) for next-token prediction.
  Depends on: TASK-002
- [ ] TASK-011: Cross-entropy loss + optimizer (SGD first, then AdamW) built
  on the autodiff engine.
  Depends on: TASK-004
- [ ] TASK-012: Training loop — wires TASK-009/010/011 together: forward
  pass, loss, backward pass, optimizer step, checkpointing (save/load model
  weights), basic logging of loss over time.
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
