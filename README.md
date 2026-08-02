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
- `src/Tensor/` — the N-dimensional `Tensor` type (see PLAN.md/TASK.md).
- `tests/Tokeniser.Tests/` — xUnit tests covering training, roundtrip encode/decode, and save/load.
- `tests/Tensor.Tests/` — xUnit tests covering tensor construction, elementwise ops, broadcasting, matmul, transpose, and reductions.
