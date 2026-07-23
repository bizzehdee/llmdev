# Tokeniser

A from-scratch byte-level Byte-Pair Encoding (BPE) tokeniser, written in C#/.NET
with no ML or tokenisation libraries. It's the first step in a personal project
to learn how LLMs work by building the pieces by hand.

## What it does

1. Reads one or more plain text files.
2. Starts from a base vocabulary of the 256 possible byte values.
3. Repeatedly finds the most frequent adjacent pair of tokens in the training
   text and merges it into a new token, until a target vocabulary size is
   reached (this is the same idea GPT-2/GPT-3 style tokenisers use).
4. Uses the learned merges to encode new text into token IDs, and to decode
   token IDs back into text.
5. Can save the trained vocabulary + merge rules to disk and reload them later.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Usage

```bash
cd src/Tokeniser
dotnet run -- <vocab-size> <file1.txt> [file2.txt ...]
```

Example:

```bash
dotnet run -- 500 ../../sample.txt
```

This trains a 500-token vocabulary from `sample.txt`, prints a sample
encode/decode roundtrip, and writes the trained vocabulary to `vocab.bpe`.

## Running tests

```bash
dotnet test
```

## Project layout

- `src/Tokeniser/BpeTokeniser.cs` — the tokeniser itself (training, encode, decode, save/load).
- `src/Tokeniser/Program.cs` — command-line entry point.
- `tests/Tokeniser.Tests/` — xUnit tests covering training, roundtrip encode/decode, and save/load.

See [PLAN.md](PLAN.md) for project scope and future direction.
