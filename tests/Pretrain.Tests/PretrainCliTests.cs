using Tokeniser;
using Training;
using Xunit;

namespace Pretrain.Tests;

public class PretrainCliTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "pretrain-tests-scratch");
    private static readonly string VocabPath = Path.Combine(ScratchDirectory, "fixture-vocab.bpe");
    private static readonly string CorpusPath = Path.Combine(ScratchDirectory, "fixture-corpus.txt");
    private static readonly string SecondCorpusPath = Path.Combine(ScratchDirectory, "fixture-corpus-2.txt");
    private static readonly string MismatchedVocabPath = Path.Combine(ScratchDirectory, "fixture-vocab-mismatched.bpe");

    static PretrainCliTests()
    {
        Directory.CreateDirectory(ScratchDirectory);

        File.WriteAllText(CorpusPath, string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 60)));
        File.WriteAllText(SecondCorpusPath, string.Concat(Enumerable.Repeat("she sells seashells by the seashore today. ", 60)));

        var tokeniser = new BpeTokeniser();
        tokeniser.Train([CorpusPath], targetVocabSize: 260, ScratchDirectory);
        tokeniser.Save(VocabPath);

        var mismatchedTokeniser = new BpeTokeniser();
        mismatchedTokeniser.Train([CorpusPath], targetVocabSize: 256, ScratchDirectory);
        mismatchedTokeniser.Save(MismatchedVocabPath);
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

    // TASK-033: --gpu.

    [Fact]
    public void Run_GpuAndOptimisedFlagsTogether_ReturnsClearError()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", CorpusPath, "--optimised", "--gpu");

        Assert.Equal(1, exitCode);
        Assert.Contains("Specify either --optimised or --gpu, not both", stderr);
    }

    [Fact]
    public void Run_GpuAllowCpuFallbackWithoutGpuFlag_ReturnsClearError()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", CorpusPath, "--gpu-allow-cpu-fallback");

        Assert.Equal(1, exitCode);
        Assert.Contains("--gpu-allow-cpu-fallback only makes sense together with --gpu", stderr);
    }

    [Fact]
    public void Run_GpuFlagWithoutCpuFallback_MatchesGpuContextsOwnAcceleratorDetection()
    {
        // Doesn't hardcode "must fail" - that would assume no real GPU is
        // ever present, which isn't a safe assumption to bake into a test.
        // Instead: ask GpuContext directly what it would decide, then prove
        // the CLI's --gpu (no fallback) behaves consistently with that -
        // on this dev machine (no working GPU driver, see TASK-031), that
        // means proving the CLI's "no GPU found" error path actually fires
        // end to end, not just in GpuContextTests' unit-level coverage.
        bool expectFailure;
        try
        {
            Tensor.GpuContext.GetAccelerator(allowCpuFallback: false);
            expectFailure = false;
        }
        catch (InvalidOperationException)
        {
            expectFailure = true;
        }
        finally
        {
            Tensor.GpuContext.Shutdown();
        }

        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        try
        {
            var (exitCode, _, stderr) = Run(
                VocabPath, checkpointPath, CorpusPath,
                "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
                "--steps", "2", "--batch-size", "2", "--gpu");

            if (expectFailure)
            {
                Assert.Equal(1, exitCode);
                Assert.Contains("No GPU accelerator", stderr);
            }
            else
            {
                Assert.Equal(0, exitCode);
            }
        }
        finally
        {
            Tensor.Tensor.Backend = Tensor.TensorBackend.Scalar;
            Tensor.GpuContext.Shutdown();
        }
    }

    [Fact]
    public void Run_GpuFlagWithCpuFallbackAllowed_SelectsGpuTensorBackendAndSucceeds()
    {
        // allowCpuFallback: true makes this succeed on any machine, real
        // GPU or not - proving the CLI wiring end to end (flag parsing,
        // GpuContext preflight, backend selection, a real training run
        // that actually uses TensorBackend.Gpu matmuls) without assuming
        // this machine has a working GPU driver.
        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        try
        {
            var (exitCode, stdout, stderr) = Run(
                VocabPath, checkpointPath, CorpusPath,
                "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
                "--steps", "2", "--batch-size", "2", "--gpu", "--gpu-allow-cpu-fallback");

            Assert.Equal(0, exitCode);
            Assert.Empty(stderr);
            Assert.Equal(Tensor.TensorBackend.Gpu, Tensor.Tensor.Backend);
            Assert.True(File.Exists(checkpointPath));
        }
        finally
        {
            Tensor.Tensor.Backend = Tensor.TensorBackend.Scalar;
            Tensor.GpuContext.Shutdown();
        }
    }

    // TASK-036: --gpu-resident-weights.

    [Fact]
    public void Run_GpuResidentWeightsWithoutGpuFlag_ReturnsClearError()
    {
        var (exitCode, _, stderr) = Run(VocabPath, "out.checkpoint", CorpusPath, "--gpu-resident-weights");

        Assert.Equal(1, exitCode);
        Assert.Contains("--gpu-resident-weights only makes sense together with --gpu", stderr);
    }

    [Fact]
    public void Run_GpuResidentWeightsFlag_TrainsCorrectlyWithParametersMovedOnToTheGpu()
    {
        // Deliberately the smallest fixture model/step count used anywhere
        // in this file: --gpu-resident-weights moves every parameter's
        // backward-pass Transpose (and the optimizer's per-parameter
        // update) onto the slow, per-element GpuFloatBuffer indexer path
        // (TASK-035 only taught matmul to avoid that, not backward or
        // AdamW) - real, measured wall-clock during development showed
        // this is dramatically slower, not faster, at this project's toy
        // scale (see README.md stage 11). This test only needs to prove
        // correctness, not speed, so it stays as small as the existing
        // --gpu fixture already is.
        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        try
        {
            var (exitCode, stdout, stderr) = Run(
                VocabPath, checkpointPath, CorpusPath,
                "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
                "--steps", "2", "--batch-size", "2", "--gpu", "--gpu-resident-weights");

            Assert.Equal(0, exitCode);
            Assert.Empty(stderr);
            Assert.True(File.Exists(checkpointPath));

            var losses = stdout.Split('\n')
                .Where(line => line.StartsWith("step ", StringComparison.Ordinal))
                .Select(line => float.Parse(line.Split("loss ")[1]))
                .ToList();
            Assert.Equal(2, losses.Count);
            Assert.All(losses, loss => Assert.False(float.IsNaN(loss) || float.IsInfinity(loss)));
        }
        finally
        {
            Tensor.Tensor.Backend = Tensor.TensorBackend.Scalar;
            Tensor.GpuContext.Shutdown();
        }
    }

    // TASK-040: --resume-from-checkpoint.

    [Fact]
    public void Run_ResumeFromCheckpoint_ContinuesTrainingKeepingTheOriginalArchitecture()
    {
        string firstCheckpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        string resumedCheckpointPath = Path.Combine(ScratchDirectory, $"resumed-{Guid.NewGuid():N}.checkpoint");

        var first = Run(
            VocabPath, firstCheckpointPath, CorpusPath,
            "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
            "--steps", "20", "--batch-size", "4", "--learning-rate", "0.01", "--weight-decay", "0",
            "--scratch-dir", ScratchDirectory);
        Assert.Equal(0, first.exitCode);

        var resumed = Run(
            VocabPath, resumedCheckpointPath, SecondCorpusPath,
            "--resume-from-checkpoint", firstCheckpointPath,
            "--steps", "20", "--batch-size", "4", "--learning-rate", "0.01", "--weight-decay", "0",
            "--scratch-dir", ScratchDirectory);

        Assert.Equal(0, resumed.exitCode);
        Assert.Empty(resumed.stderr);
        Assert.True(File.Exists(resumedCheckpointPath));

        var original = ModelCheckpoint.Load(firstCheckpointPath);
        var continued = ModelCheckpoint.Load(resumedCheckpointPath);
        Assert.Equal(original.EmbeddingDim, continued.EmbeddingDim);
        Assert.Equal(original.NumLayers, continued.NumLayers);
        Assert.Equal(original.NumHeads, continued.NumHeads);
        Assert.Equal(original.MaxSequenceLength, continued.MaxSequenceLength);

        // Weights should have moved from where the first run left them -
        // proof this actually continued training rather than silently
        // starting fresh and discarding the resumed architecture's values.
        var originalFirstParam = original.Parameters()[0].Value.ToArray();
        var continuedFirstParam = continued.Parameters()[0].Value.ToArray();
        Assert.NotEqual(originalFirstParam, continuedFirstParam);
    }

    [Fact]
    public void Run_ResumeFromCheckpointWithArchitectureFlag_ReturnsClearError()
    {
        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        var setup = Run(
            VocabPath, checkpointPath, CorpusPath,
            "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
            "--steps", "2", "--batch-size", "2");
        Assert.Equal(0, setup.exitCode);

        var (exitCode, _, stderr) = Run(
            VocabPath, "out.checkpoint", CorpusPath,
            "--resume-from-checkpoint", checkpointPath, "--embedding-dim", "8");

        Assert.Equal(1, exitCode);
        Assert.Contains("cannot be combined with --resume-from-checkpoint", stderr);
    }

    [Fact]
    public void Run_ResumeFromCheckpointWithMismatchedVocab_ReturnsClearError()
    {
        string checkpointPath = Path.Combine(ScratchDirectory, $"trained-{Guid.NewGuid():N}.checkpoint");
        var setup = Run(
            VocabPath, checkpointPath, CorpusPath,
            "--embedding-dim", "8", "--layers", "1", "--heads", "2", "--context-length", "8",
            "--steps", "2", "--batch-size", "2");
        Assert.Equal(0, setup.exitCode);

        var (exitCode, _, stderr) = Run(
            MismatchedVocabPath, "out.checkpoint", CorpusPath,
            "--resume-from-checkpoint", checkpointPath);

        Assert.Equal(1, exitCode);
        Assert.Contains("vocab size", stderr);
    }

    [Fact]
    public void Run_ResumeFromNonexistentCheckpoint_ReturnsClearError()
    {
        var (exitCode, _, stderr) = Run(
            VocabPath, "out.checkpoint", CorpusPath,
            "--resume-from-checkpoint", "/nonexistent/checkpoint.bin");

        Assert.Equal(1, exitCode);
        Assert.Contains("Failed to load checkpoint to resume", stderr);
    }
}
