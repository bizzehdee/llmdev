using System.Text.Json;
using Tokeniser;

namespace Training;

/// <summary>
/// Loads instruction/response pairs for TASK-016's instruction tuning
/// (SFT) from a JSON Lines file - one <c>{"instruction": "...", "response": "..."}</c>
/// object per line. JSON Lines chosen over a plain-text template
/// (PLAN.md stage 10's two candidate formats) as the more extensible of
/// the two - a plain delimiter-based template would need escaping rules
/// of its own the moment an instruction or response contains the
/// delimiter itself, which JSON already handles.
/// </summary>
public static class SftDataset
{
    private const string PromptTemplate = "### Instruction:\n{0}\n\n### Response:\n";

    /// <summary>
    /// The literal marker that opens every templated prompt. Exposed so
    /// callers that generate *from* a fine-tuned model (e.g. the chat CLI's
    /// instruction-tuned mode, TASK-027) can detect where a response has
    /// run on into a hallucinated next turn, without duplicating the
    /// template string themselves.
    /// </summary>
    public const string InstructionMarker = "### Instruction:";

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    /// <summary>Wraps <paramref name="instruction"/> in the same prompt template <see cref="Tokenize"/> uses, so callers never duplicate the template string.</summary>
    public static string FormatPrompt(string instruction) => string.Format(PromptTemplate, instruction);

    /// <summary>Reads every non-blank line as one JSON-encoded <see cref="SftExample"/>, tokenising each via <paramref name="tokeniser"/>.</summary>
    public static IReadOnlyList<SftTokenizedExample> Load(string path, BpeTokeniser tokeniser)
    {
        var examples = new List<SftTokenizedExample>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var example = JsonSerializer.Deserialize<SftExample>(line, JsonOptions)
                ?? throw new InvalidOperationException($"Malformed SFT example (deserialised to null): {line}");
            examples.Add(Tokenize(example, tokeniser));
        }
        return examples;
    }

    /// <summary>
    /// Wraps the instruction in a fixed prompt template and tokenises the
    /// templated prompt and the response *separately* (not the whole
    /// templated string as one call) - guaranteeing an exact token-level
    /// split between them regardless of where <see cref="BpeTokeniser"/>'s
    /// merges would otherwise fall if asked to tokenise across that
    /// boundary in one pass. Anything that later prompts a fine-tuned
    /// model (e.g. a chat CLI) must format its prompt with this same
    /// template, or the model will be seeing text shaped differently than
    /// what it was tuned on.
    /// </summary>
    public static SftTokenizedExample Tokenize(SftExample example, BpeTokeniser tokeniser)
    {
        var promptIds = tokeniser.Encode(FormatPrompt(example.Instruction));
        var responseIds = tokeniser.Encode(example.Response);

        var allIds = promptIds.Concat(responseIds).ToArray();
        var inputIds = allIds[..^1];
        var targetIds = allIds[1..];

        // Position i's target is allIds[i + 1] (the standard next-token
        // shift); that target falls in the response iff i + 1 is at or
        // past where the prompt's tokens end.
        var responseMask = new bool[targetIds.Length];
        for (int i = 0; i < responseMask.Length; i++)
        {
            responseMask[i] = i + 1 >= promptIds.Count;
        }

        return new SftTokenizedExample(inputIds, targetIds, responseMask);
    }
}
