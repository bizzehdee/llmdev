using Generation;
using Model;
using Tensor;
using Tokeniser;
using Training;

namespace Chat;

/// <summary>
/// The chat CLI's actual logic, factored out of <c>Program.cs</c> the same
/// way <c>Tokeniser.TokeniserCli</c> is - top-level statements can't be
/// invoked from a test project, and I/O streams are taken as parameters so
/// tests can script a conversation and capture output without touching the
/// real console.
///
/// Loads a trained <see cref="GptModel"/> checkpoint and tokeniser vocab,
/// then loops: read a line of input, generate a continuation, print it,
/// repeat. Conversation state is a single growing token-id sequence (not
/// re-encoded text each turn) fed back into <see cref="TextGenerator"/>
/// every turn, so multi-turn context accumulates correctly - including
/// through its existing sliding-window handling once the conversation
/// exceeds the model's context length.
///
/// This is a *raw next-token-prediction model*, not an instruction-tuned
/// assistant (see PLAN.md stage 8) - it continues text in the style of
/// whatever it was trained on, and the banner below says so up front
/// rather than overselling "chatbot".
/// </summary>
public static class ChatCli
{
    private const string ExitCommand = "/exit";

    public static int Run(string[] args, TextReader stdin, TextWriter stdout, TextWriter stderr)
    {
        if (args.Length < 2)
        {
            stdout.WriteLine("Usage: Chat <checkpoint-path> <vocab-path> [--temperature <f>] [--top-k <n>] [--top-p <f>] [--max-new-tokens <n>] [--context-length <n>] [--instruction-tuned] [--optimised]");
            stdout.WriteLine("Loads a trained model checkpoint and tokeniser vocabulary, then lets you converse with it turn by turn.");
            stdout.WriteLine($"Type {ExitCommand} (or send EOF, e.g. Ctrl+D) to leave.");
            stdout.WriteLine();
            stdout.WriteLine("This is a raw next-token-prediction model, not an instruction-tuned assistant: it will");
            stdout.WriteLine("continue text in the style of whatever it was trained on, not necessarily answer questions");
            stdout.WriteLine("or follow instructions, unless the training corpus was itself shaped like dialogue, or");
            stdout.WriteLine("--instruction-tuned is passed for a checkpoint that went through instruction tuning (see the Sft CLI).");
            return 1;
        }

        string checkpointPath = args[0];
        string vocabPath = args[1];

        var options = new SamplingOptions();
        int maxNewTokens = 50;
        int? contextLengthOption = null;
        bool instructionTuned = false;
        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--temperature" when i + 1 < args.Length && float.TryParse(args[i + 1], out float temperature):
                    options = options with { Temperature = temperature };
                    i++;
                    break;
                case "--top-k" when i + 1 < args.Length && int.TryParse(args[i + 1], out int topK):
                    options = options with { TopK = topK };
                    i++;
                    break;
                case "--top-p" when i + 1 < args.Length && float.TryParse(args[i + 1], out float topP):
                    options = options with { TopP = topP };
                    i++;
                    break;
                case "--max-new-tokens" when i + 1 < args.Length && int.TryParse(args[i + 1], out int requestedMaxNewTokens):
                    maxNewTokens = requestedMaxNewTokens;
                    i++;
                    break;
                case "--context-length" when i + 1 < args.Length && int.TryParse(args[i + 1], out int requestedContextLength) && requestedContextLength > 0:
                    contextLengthOption = requestedContextLength;
                    i++;
                    break;
                case "--instruction-tuned":
                    // TASK-027: opt-in, so a purely-pretrained checkpoint's
                    // existing raw-continuation behaviour isn't disturbed.
                    instructionTuned = true;
                    break;
                case "--optimised":
                    // TASK-015: opt into the TensorPrimitives-backed matmul fast
                    // path for everything downstream in this call chain (see
                    // Tensor.Backend's doc comment for why AsyncLocal, not a
                    // plain static). Always safe to select: ops that can't use
                    // it (e.g. a disk-backed tensor) fall back to the scalar
                    // implementation transparently.
                    Tensor.Tensor.Backend = TensorBackend.Optimised;
                    break;
                default:
                    stderr.WriteLine($"Unrecognised or malformed option: {args[i]}");
                    return 1;
            }
        }

        GptModel model;
        BpeTokeniser tokeniser;
        try
        {
            model = ModelCheckpoint.Load(checkpointPath);
            tokeniser = BpeTokeniser.Load(vocabPath);
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException)
        {
            // IOException: missing/unreadable file (a user-supplied path,
            // so this is a real system boundary, not a "can't happen" case).
            // InvalidOperationException: ModelCheckpoint.Load's own
            // validation of a corrupted/mismatched checkpoint file.
            stderr.WriteLine($"Failed to load checkpoint or vocabulary: {ex.Message}");
            return 1;
        }

        // TASK-028: an adjustable window, independent of just waiting for
        // the model's own fixed MaxSequenceLength to be reached - it must
        // never exceed that real limit, so this is validated rather than
        // silently clamped.
        int contextLength = contextLengthOption ?? model.MaxSequenceLength;
        if (contextLength > model.MaxSequenceLength)
        {
            stderr.WriteLine($"--context-length {contextLength} exceeds this model's MaxSequenceLength ({model.MaxSequenceLength}).");
            return 1;
        }

        stdout.WriteLine(instructionTuned
            ? "Loaded model and vocabulary. Instruction-tuned mode: each turn is wrapped in the same"
            : "Loaded model and vocabulary. This is a raw next-token-prediction model, not an");
        stdout.WriteLine(instructionTuned
            ? "prompt template the model was fine-tuned on, and generation stops at the next"
            : "instruction-tuned assistant: it will continue text in the style of whatever it was");
        stdout.WriteLine(instructionTuned
            ? "turn boundary instead of running on."
            : "trained on, not necessarily answer questions or follow instructions.");
        stdout.WriteLine($"Type {ExitCommand} (or send EOF, e.g. Ctrl+D) to leave.");
        stdout.WriteLine();

        var conversationTokenIds = new List<int>();
        var random = new Random();

        while (true)
        {
            stdout.Write("> ");
            string? line = stdin.ReadLine();
            if (line is null || line == ExitCommand)
            {
                stdout.WriteLine("Goodbye.");
                return 0;
            }

            conversationTokenIds.AddRange(tokeniser.Encode(instructionTuned ? SftDataset.FormatPrompt(line) : line));
            conversationTokenIds = TruncateToContextLength(conversationTokenIds, contextLength);

            var extended = instructionTuned
                ? TextGenerator.GenerateTokenIdsUntilStopSequence(model, tokeniser, conversationTokenIds.ToArray(), maxNewTokens, SftDataset.InstructionMarker, options, random)
                : TextGenerator.GenerateTokenIds(model, conversationTokenIds.ToArray(), maxNewTokens, options, random);
            var newTokenIds = extended.Skip(conversationTokenIds.Count).ToList();
            conversationTokenIds = TruncateToContextLength(extended, contextLength);

            stdout.WriteLine(tokeniser.Decode(newTokenIds));
        }
    }

    /// <summary>Keeps only the most recent <paramref name="contextLength"/> tokens - the CLI's own, potentially tighter, cap ahead of the model's own <c>MaxSequenceLength</c> sliding window in <see cref="TextGenerator"/>.</summary>
    private static List<int> TruncateToContextLength(List<int> tokenIds, int contextLength)
    {
        return tokenIds.Count > contextLength
            ? tokenIds.Skip(tokenIds.Count - contextLength).ToList()
            : tokenIds;
    }
}
