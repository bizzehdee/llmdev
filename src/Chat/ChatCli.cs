using Generation;
using Model;
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
            stdout.WriteLine("Usage: Chat <checkpoint-path> <vocab-path> [--temperature <f>] [--top-k <n>] [--top-p <f>] [--max-new-tokens <n>]");
            stdout.WriteLine("Loads a trained model checkpoint and tokeniser vocabulary, then lets you converse with it turn by turn.");
            stdout.WriteLine($"Type {ExitCommand} (or send EOF, e.g. Ctrl+D) to leave.");
            stdout.WriteLine();
            stdout.WriteLine("This is a raw next-token-prediction model, not an instruction-tuned assistant: it will");
            stdout.WriteLine("continue text in the style of whatever it was trained on, not necessarily answer questions");
            stdout.WriteLine("or follow instructions, unless the training corpus was itself shaped like dialogue.");
            return 1;
        }

        string checkpointPath = args[0];
        string vocabPath = args[1];

        var options = new SamplingOptions();
        int maxNewTokens = 50;
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

        stdout.WriteLine("Loaded model and vocabulary. This is a raw next-token-prediction model, not an");
        stdout.WriteLine("instruction-tuned assistant: it will continue text in the style of whatever it was");
        stdout.WriteLine("trained on, not necessarily answer questions or follow instructions.");
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

            conversationTokenIds.AddRange(tokeniser.Encode(line));

            var extended = TextGenerator.GenerateTokenIds(model, conversationTokenIds.ToArray(), maxNewTokens, options, random);
            var newTokenIds = extended.Skip(conversationTokenIds.Count).ToList();
            conversationTokenIds = extended;

            stdout.WriteLine(tokeniser.Decode(newTokenIds));
        }
    }
}
