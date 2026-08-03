using Model;
using Tokeniser;
using Training;
using Xunit;

namespace Chat.Tests;

public class ChatCliTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "chat-tests-scratch");
    private static readonly string CheckpointPath = Path.Combine(ScratchDirectory, "fixture-model.checkpoint");
    private static readonly string VocabPath = Path.Combine(ScratchDirectory, "fixture-vocab.bpe");

    static ChatCliTests()
    {
        Directory.CreateDirectory(ScratchDirectory);

        string corpusPath = Path.Combine(ScratchDirectory, "fixture-corpus.txt");
        File.WriteAllText(corpusPath, string.Concat(Enumerable.Repeat("hello there how are you today. i am fine thank you. ", 30)));

        var tokeniser = new BpeTokeniser();
        tokeniser.Train([corpusPath], targetVocabSize: 260, ScratchDirectory);
        tokeniser.Save(VocabPath);

        var model = new GptModel(vocabSize: tokeniser.VocabSize, embeddingDim: 16, numLayers: 2, numHeads: 2, maxSequenceLength: 32, random: new Random(1));
        ModelCheckpoint.Save(model, CheckpointPath);

        File.Delete(corpusPath);
    }

    private static (int exitCode, string stdout, string stderr) Run(string input, params string[] args)
    {
        using var stdin = new StringReader(input);
        using var stdout = new StringWriter();
        using var stderr = new StringWriter();
        int exitCode = ChatCli.Run(args, stdin, stdout, stderr);
        return (exitCode, stdout.ToString(), stderr.ToString());
    }

    [Fact]
    public void Run_TooFewArguments_PrintsUsageAndReturnsOne()
    {
        var (exitCode, stdout, _) = Run("", CheckpointPath);

        Assert.Equal(1, exitCode);
        Assert.Contains("Usage: Chat", stdout);
    }

    [Fact]
    public void Run_UsageMessage_IsHonestAboutNotBeingInstructionTuned()
    {
        var (_, stdout, _) = Run("");

        Assert.Contains("not an instruction-tuned assistant", stdout);
    }

    [Fact]
    public void Run_UnrecognisedFlag_ReturnsError()
    {
        var (exitCode, _, stderr) = Run("", CheckpointPath, VocabPath, "--not-a-real-flag");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_MalformedFlagValue_ReturnsError()
    {
        var (exitCode, _, stderr) = Run("", CheckpointPath, VocabPath, "--temperature", "not-a-number");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_FlagMissingValue_ReturnsError()
    {
        var (exitCode, _, stderr) = Run("", CheckpointPath, VocabPath, "--temperature");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_MissingCheckpointFile_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run("", "/nonexistent/model.checkpoint", VocabPath);

        Assert.Equal(1, exitCode);
        Assert.Contains("Failed to load checkpoint or vocabulary", stderr);
    }

    [Fact]
    public void Run_MissingVocabFile_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run("", CheckpointPath, "/nonexistent/vocab.bpe");

        Assert.Equal(1, exitCode);
        Assert.Contains("Failed to load checkpoint or vocabulary", stderr);
    }

    [Fact]
    public void Run_CorruptedCheckpointFile_ReturnsErrorWithMessage()
    {
        string corruptPath = Path.Combine(ScratchDirectory, $"corrupt-{Guid.NewGuid():N}.checkpoint");
        var bytes = File.ReadAllBytes(CheckpointPath);
        const int headerInts = 6;
        int countOffset = headerInts * sizeof(int);
        BitConverter.GetBytes(999).CopyTo(bytes, countOffset);
        File.WriteAllBytes(corruptPath, bytes);

        try
        {
            var (exitCode, _, stderr) = Run("", corruptPath, VocabPath);

            Assert.Equal(1, exitCode);
            Assert.Contains("Failed to load checkpoint or vocabulary", stderr);
        }
        finally
        {
            File.Delete(corruptPath);
        }
    }

    [Fact]
    public void Run_ExitCommand_PrintsGoodbyeAndReturnsZero()
    {
        var (exitCode, stdout, _) = Run("/exit\n", CheckpointPath, VocabPath);

        Assert.Equal(0, exitCode);
        Assert.Contains("Goodbye.", stdout);
    }

    [Fact]
    public void Run_EndOfInputWithoutExitCommand_EndsGracefully()
    {
        var (exitCode, stdout, _) = Run("", CheckpointPath, VocabPath); // no input at all -> immediate EOF

        Assert.Equal(0, exitCode);
        Assert.Contains("Goodbye.", stdout);
    }

    [Fact]
    public void Run_SingleTurnConversation_PrintsAPromptAndAResponse()
    {
        var (exitCode, stdout, _) = Run("hello there\n/exit\n", CheckpointPath, VocabPath, "--max-new-tokens", "5");

        Assert.Equal(0, exitCode);
        Assert.Contains("> ", stdout);
        Assert.Contains("Goodbye.", stdout);
    }

    [Fact]
    public void Run_MultiTurnConversation_CompletesAllTurnsWithoutError()
    {
        // Generated text is effectively random with this untrained fixture
        // model, so it could coincidentally contain "> " itself - this
        // checks the conversation runs to completion across several turns
        // without throwing, rather than counting prompt markers in output
        // that isn't otherwise constrained.
        var (exitCode, stdout, _) = Run("hello there\nhow are you\nwhat now\n/exit\n", CheckpointPath, VocabPath, "--max-new-tokens", "5");

        Assert.Equal(0, exitCode);
        Assert.Contains("Goodbye.", stdout);
    }

    [Fact]
    public void Run_WithTemperatureTopKTopPAndMaxNewTokensFlags_StillWorks()
    {
        var (exitCode, stdout, _) = Run("hello\n/exit\n", CheckpointPath, VocabPath,
            "--temperature", "0.7", "--top-k", "10", "--top-p", "0.9", "--max-new-tokens", "5");

        Assert.Equal(0, exitCode);
        Assert.Contains("Goodbye.", stdout);
    }

    [Fact]
    public void Run_GreedyTemperatureZero_IsDeterministicAcrossRuns()
    {
        var (_, stdoutFirst, _) = Run("hello there\n/exit\n", CheckpointPath, VocabPath, "--temperature", "0", "--max-new-tokens", "8");
        var (_, stdoutSecond, _) = Run("hello there\n/exit\n", CheckpointPath, VocabPath, "--temperature", "0", "--max-new-tokens", "8");

        Assert.Equal(stdoutFirst, stdoutSecond);
    }
}
