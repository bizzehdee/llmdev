# llmdev

A personal project to learn how LLMs work by building one from first
principles in C#/.NET — no ML, tokenisation, or tensor/autodiff libraries.
Every mechanism (tokeniser, tensor math, autodiff, attention, training loop)
is written by hand so it's understood, not a black box behind a library call.

See [PLAN.md](PLAN.md) for the full roadmap and [TASK.md](TASK.md) for
current progress.

## Tokeniser (done)

A from-scratch byte-level Byte-Pair Encoding (BPE) tokeniser.

1. Reads one or more plain text files, or a directory of `.txt` files
   (non-`.txt` files, e.g. `.epub`, are skipped).
2. Starts from a base vocabulary of the 256 possible byte values.
3. Repeatedly finds the most frequent adjacent pair of tokens in the training
   text and merges it into a new token, until a target vocabulary size is
   reached (this is the same idea GPT-2/GPT-3 style tokenisers use).
4. Uses the learned merges to encode new text into token IDs, and to decode
   token IDs back into text.
5. Can save the trained vocabulary + merge rules to disk and reload them later.

Training data for a large corpus doesn't need to fit in RAM: token state is
held in memory-mapped scratch files on disk rather than the managed heap, so
the OS can reclaim it under memory pressure instead of risking an OOM kill.

### Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

### Usage

```bash
cd src/Tokeniser
dotnet run -- <vocab-size> <file-or-directory> [file-or-directory ...] [--scratch-dir <dir>]
```

Examples:

```bash
# Train on a single file
dotnet run -- 500 ../../sample.txt

# Train on every .txt file in a directory
dotnet run -- 2000 ~/Documents/some-text-corpus
```

This trains a vocabulary from the given input(s), prints a sample
encode/decode roundtrip, and writes the trained vocabulary to `vocab.bpe`.
Before training starts, estimated disk scratch usage is checked against
available disk space and refused if unsafe (and refuses a `tmpfs`-backed
scratch directory, e.g. `/tmp` on many Linux distros, since that's RAM under
a different name).

### Running tests

```bash
dotnet test
```

### Project layout

- `src/Tokeniser/BpeTokeniser.cs` — the tokeniser itself (training, encode, decode, save/load).
- `src/Tokeniser/Program.cs` — command-line entry point.
- `src/Common/MappedArray.cs` — disk-backed (memory-mapped) array type shared by the tokeniser and the tensor engine.
- `src/Tensor/` — the N-dimensional `Tensor` type and the `Variable` reverse-mode autodiff engine built on it (see PLAN.md/TASK.md).
- `src/Model/` — learned model components built on `Tensor`/`Variable`: token/positional embeddings, scaled dot-product and multi-head (causal) attention, layernorm, feed-forward, a GPT-2-style pre-norm `TransformerBlock`, and `GptModel` — the full decoder-only model (stacked blocks + weight-tied output projection to vocabulary logits).
- `tests/Tokeniser.Tests/` — xUnit tests covering training, roundtrip encode/decode, and save/load.
- `tests/Tensor.Tests/` — xUnit tests covering tensor construction, elementwise ops, broadcasting, matmul, transpose, reductions, and autodiff (finite-difference gradient checks).
- `tests/Model.Tests/` — xUnit tests covering model components (embeddings, attention, layernorm, feed-forward, transformer block, and the full `GptModel`), with finite-difference gradient checks and end-to-end causal-masking correctness throughout.
- `src/Training/` — `TokenCorpus` (a disk-backed token-id stream), `BatchSampler` (fixed-length next-token-prediction input/target windows), `CrossEntropyLoss`, `SgdOptimizer`/`AdamWOptimizer`, `ModelCheckpoint` (binary save/load of a `GptModel`'s architecture + weights), and `Trainer` — the training loop that wires all of it together.
- `tests/Training.Tests/` — xUnit tests covering corpus storage, batch sampling correctness, loss values/gradients, optimizer convergence, checkpoint round-tripping, and a genuine end-to-end training run (loss measurably drops on a small repetitive corpus).
- `src/Generation/` — `SamplingOptions`, `TokenSampler` (greedy/temperature/top-k/top-p), and `TextGenerator` — runs a `GptModel` autoregressively and decodes the result back to text via the tokeniser.
- `tests/Generation.Tests/` — xUnit tests covering sampling correctness (deterministic greedy, statistical distribution checks) and end-to-end generation.

## Project status

All 7 planned stages are done (see PLAN.md/TASK.md): tokeniser, tensor +
autodiff engine, embeddings, attention + transformer block, full model
assembly, training loop, and generation. The project can train a small
GPT-style model from scratch on a text corpus and generate text from it,
entirely from first principles. See PLAN.md's "Known limitations /
deferred" section for documented trade-offs (not bugs) a larger training
run might need to revisit.
