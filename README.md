# llmdev

A personal project to learn how LLMs work by building one from first
principles in C#/.NET — no ML, tokenisation, or tensor/autodiff libraries.
Every mechanism (tokeniser, tensor math, autodiff, attention, training loop)
is written by hand so it's understood, not a black box behind a library call.

See [PLAN.md](PLAN.md) for the full roadmap and [TASK.md](TASK.md) for
current progress.

## Full workflow: text → tokeniser → trained model → generated text

There's a command-line tool for the tokeniser (step 1), but training and
generation (steps 2–5) are library-only so far — no unified CLI exists yet.
The example below is a small C# program that walks through the whole
pipeline; paste it into a console project that references the `Tokeniser`,
`Model`, `Training`, and `Generation` projects (see
[Requirements](#requirements) and [Running the example](#running-the-example)
below).

### 1. Train a tokeniser (CLI)

See [Tokeniser usage](#usage) below. This produces a `vocab.bpe` file.

### 2–5. Train a model and generate text (code)

```csharp
using Model;
using Tokeniser;
using Training;
using Generation;

const string ScratchDirectory = "/home/you/.cache/llmdev-scratch"; // real disk, not /tmp - see Tokeniser usage notes below
const string CorpusPath = "corpus.txt";

// 2. Train (or load) a tokeniser, then encode the training corpus into a
//    disk-backed token stream and a batch sampler over it.
var tokeniser = new BpeTokeniser();
tokeniser.Train([CorpusPath], targetVocabSize: 2000, ScratchDirectory);
tokeniser.Save("vocab.bpe");

var tokenIds = tokeniser.Encode(File.ReadAllText(CorpusPath));
using var corpus = new TokenCorpus(tokenIds, ScratchDirectory);
var sampler = new BatchSampler(corpus, contextLength: 64);

// 3. Build a model and train it.
var model = new GptModel(
    vocabSize: tokeniser.VocabSize,
    embeddingDim: 128,
    numLayers: 4,
    numHeads: 4,
    maxSequenceLength: 64);

var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: 3e-4f);
var trainer = new Trainer(model, sampler, optimizer);

trainer.Run(steps: 2000, batchSize: 8, onStep: (step, loss) =>
{
    if (step % 100 == 0)
    {
        Console.WriteLine($"step {step}: loss {loss:F4}");
    }
});

// 4. Checkpoint the trained model.
ModelCheckpoint.Save(model, "model.checkpoint");

// 5. Reload (possibly in a separate run/process) and generate text.
var loadedModel = ModelCheckpoint.Load("model.checkpoint");
var loadedTokeniser = BpeTokeniser.Load("vocab.bpe");

string generated = TextGenerator.Generate(
    loadedModel,
    loadedTokeniser,
    prompt: "Once upon a time",
    maxNewTokens: 100,
    options: new SamplingOptions { Temperature = 0.8f, TopK = 40 });

Console.WriteLine(generated);
```

Notes on the numbers above: `embeddingDim`/`numLayers`/`numHeads`/`steps`
are small, fast-to-run placeholders, not tuned hyperparameters — scale them
up (and expect training to take much longer) for a model that produces
coherent text. `contextLength`/`maxSequenceLength` must match between the
sampler and the model. See PLAN.md's "Known limitations / deferred" section
for trade-offs (e.g. no KV-cache, so generation gets slower per token as
the sequence grows) worth knowing about before scaling this up.

### Running the example

```bash
dotnet new console -o MyTraining
cd MyTraining
dotnet add reference ../src/Tokeniser/Tokeniser.csproj ../src/Model/Model.csproj ../src/Training/Training.csproj ../src/Generation/Generation.csproj
# paste the code above into Program.cs, adjust CorpusPath/ScratchDirectory
dotnet run
```

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
