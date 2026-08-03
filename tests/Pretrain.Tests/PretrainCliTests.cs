using Tokeniser;
using Training;
using Xunit;

namespace Pretrain.Tests;

public class PretrainCliTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "pretrain-tests-scratch");
    private static readonly string VocabPath = Path.Combine(ScratchDirectory, "fixture-vocab.bpe");
    private static readonly string CorpusPath = Path.Combine(ScratchDirectory, "fixture-corpus.txt");

    static PretrainCliTests()
    {
        Directory.CreateDirectory(ScratchDirectory);

        File.WriteAllText(CorpusPath, string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 60)));

        var tokeniser = new BpeTokeniser();
        tokeniser.Train([CorpusPath], targetVocabSize: 260, ScratchDirectory);
        tokeniser.Save(VocabPath);
    }

    private static (int exitCode, string stdout, string stderr) Run(params string[] args)
    {
        using var stdout = new StringWriter();
        using var stderr = new StringWriter();
        int exitCode = PretrainCli.Run(args, stdout, stderr);
        return (exitCode, stdout.ToString(), stderr.ToString());
    }

    [Fact]
    public void Run_TooFewArguments_PrintsUsageAndReturnsOne()
    {
        var (exitCode, stdout, _) = Run(VocabPath, "out.checkpoint");

        Assert.Equal(1, exitCode);
        Assert.Contains("Usage: Pretrain", stdout);
    }

    [Fact]
    public void Run_UnrecognisedFlag_ReturnsError()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", CorpusPath, "--not-a-real-flag");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_MalformedFlagValue_ReturnsError()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", CorpusPath, "--steps", "not-a-number");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_MissingVocabFile_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run("/nonexistent/vocab.bpe", "out.checkpoint", CorpusPath);

        Assert.Equal(1, exitCode);
        Assert.Contains("Failed to load tokeniser vocabulary", stderr);
    }

    [Fact]
    public void Run_MissingCorpusInput_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", "/nonexistent/corpus.txt");

        Assert.Equal(1, exitCode);
        Assert.Contains("File or directory not found", stderr);
    }

    [Fact]
    public void Run_NoCorpusPositionalArgs_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", "--steps", "5");

        Assert.Equal(1, exitCode);
        Assert.Contains("Provide at least one corpus file or directory", stderr);
    }

    [Fact]
    public void Run_EmptyCorpusDirectory_ReturnsErrorWithMessage()
    {
        string emptyDir = Path.Combine(ScratchDirectory, $"empty-{Guid.NewGuid():N}");
        Directory.CreateDirectory(emptyDir);

        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", emptyDir);

        Assert.Equal(1, exitCode);
        Assert.Contains("No .txt files found", stderr);
    }

    [Fact]
    public void Run_EndToEnd_ProducesALoadableCheckpointWithDecreasedLoss()
    {
        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");

        var (exitCode, stdout, stderr) = Run(
            VocabPath, checkpointPath, CorpusPath,
            "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
            "--steps", "60", "--batch-size", "4", "--learning-rate", "0.01", "--weight-decay", "0",
            "--scratch-dir", ScratchDirectory);

        Assert.Equal(0, exitCode);
        Assert.Empty(stderr);
        Assert.True(File.Exists(checkpointPath), "Expected a checkpoint file to be written.");

        var losses = stdout.Split('\n')
            .Where(line => line.StartsWith("step ", StringComparison.Ordinal))
            .Select(line => float.Parse(line.Split("loss ")[1]))
            .ToList();
        Assert.True(losses.Count >= 2, "Expected at least a first and last loss to be printed.");
        Assert.True(losses[^1] < losses[0], $"Expected loss to drop: {losses[0]} -> {losses[^1]}.");

        var loaded = ModelCheckpoint.Load(checkpointPath);
        Assert.Equal(8, loaded.EmbeddingDim);
        Assert.Equal(1, loaded.NumLayers);
    }

    [Fact]
    public void Run_OptimisedFlag_SelectsOptimisedTensorBackend()
    {
        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        try
        {
            var (exitCode, _, _) = Run(
                VocabPath, checkpointPath, CorpusPath,
                "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
                "--steps", "2", "--batch-size", "2", "--optimised");

            Assert.Equal(0, exitCode);
            Assert.Equal(Tensor.TensorBackend.Optimised, Tensor.Tensor.Backend);
        }
        finally
        {
            Tensor.Tensor.Backend = Tensor.TensorBackend.Scalar;
        }
    }
}
