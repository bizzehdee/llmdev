using Xunit;

namespace Tokeniser.Tests;

public class TokeniserCliTests
{
    // Real disk (not /tmp, which is tmpfs on this dev machine - confirmed
    // via `mount`), matching the production CLI's own requirement.
    private static readonly string ScratchDirectory = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "tokeniser-cli-tests-scratch");

    static TokeniserCliTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    private static (int exitCode, string stdout, string stderr) Run(params string[] args)
    {
        using var stdout = new StringWriter();
        using var stderr = new StringWriter();
        int exitCode = TokeniserCli.Run(args, stdout, stderr);
        return (exitCode, stdout.ToString(), stderr.ToString());
    }

    [Fact]
    public void Run_NoArguments_PrintsUsageAndReturnsOne()
    {
        var (exitCode, stdout, _) = Run();

        Assert.Equal(1, exitCode);
        Assert.Contains("Usage: Tokeniser", stdout);
    }

    [Fact]
    public void Run_NonNumericVocabSize_ReturnsErrorAndExplains()
    {
        var (exitCode, _, stderr) = Run("not-a-number", "somefile.txt");

        Assert.Equal(1, exitCode);
        Assert.Contains("vocab-size", stderr);
    }

    [Fact]
    public void Run_ScratchDirFlagMissingValue_ReturnsError()
    {
        var (exitCode, _, stderr) = Run("500", "somefile.txt", "--scratch-dir");

        Assert.Equal(1, exitCode);
        Assert.Contains("--scratch-dir requires a path argument", stderr);
    }

    [Fact]
    public void Run_NoPositionalInputsAfterFlags_ReturnsError()
    {
        var (exitCode, _, stderr) = Run("500", "--scratch-dir", ScratchDirectory);

        Assert.Equal(1, exitCode);
        Assert.Contains("Provide at least one input", stderr);
    }

    [Fact]
    public void Run_NonExistentFileOrDirectory_ReturnsError()
    {
        var (exitCode, _, stderr) = Run("500", "/this/path/does/not/exist.txt", "--scratch-dir", ScratchDirectory);

        Assert.Equal(1, exitCode);
        Assert.Contains("not found", stderr);
    }

    [Fact]
    public void Run_DirectoryWithNoTxtFiles_ReturnsError()
    {
        string emptyDir = Path.Combine(ScratchDirectory, $"empty-{Guid.NewGuid():N}");
        Directory.CreateDirectory(emptyDir);

        var (exitCode, _, stderr) = Run("500", emptyDir, "--scratch-dir", ScratchDirectory);

        Assert.Equal(1, exitCode);
        Assert.Contains("No .txt files found", stderr);
    }

    [Fact]
    public void Run_TmpfsScratchDirectory_IsRefused()
    {
        // This dev machine's /tmp is tmpfs (confirmed via `mount`); this
        // test is inherently tied to that environment fact, not mocked.
        var (exitCode, _, stderr) = Run("500", "somefile.txt", "--scratch-dir", "/tmp");

        Assert.Equal(1, exitCode);
        Assert.Contains("tmpfs", stderr);
    }

    [Fact]
    public void Run_ValidInputs_TrainsAndReportsSuccessfulRoundtrip()
    {
        // A target vocab size that's an exact multiple of 100, and a sample
        // long enough to exceed 200 characters (get truncated for the printed
        // sample), so this exercises both sides of the "print progress every
        // 100 merges or on the final one" check and the "truncate the printed
        // sample" check, not just the common case.
        string corpusPath = Path.Combine(ScratchDirectory, $"corpus-{Guid.NewGuid():N}.txt");
        string corpus = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. the dog barks. ", 10));
        File.WriteAllText(corpusPath, corpus);
        string workingDir = Directory.GetCurrentDirectory();
        string vocabPath = Path.Combine(workingDir, "vocab.bpe");

        try
        {
            var (exitCode, stdout, _) = Run("300", corpusPath, "--scratch-dir", ScratchDirectory);

            Assert.Equal(0, exitCode);
            Assert.Contains("Trained vocabulary size:", stdout);
            Assert.Contains("Roundtrip match: True", stdout);
            Assert.Contains("Saved vocabulary + merges to vocab.bpe", stdout);
            Assert.True(File.Exists(vocabPath));
        }
        finally
        {
            File.Delete(corpusPath);
            File.Delete(vocabPath);
        }
    }

    [Fact]
    public void Run_DirectoryInput_ExpandsToTxtFilesOnly()
    {
        string dir = Path.Combine(ScratchDirectory, $"corpus-dir-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "a.txt"), "the quick brown fox jumps over the lazy dog.");
        File.WriteAllText(Path.Combine(dir, "b.epub"), "should be ignored, not valid text anyway");
        string workingDir = Directory.GetCurrentDirectory();
        string vocabPath = Path.Combine(workingDir, "vocab.bpe");

        try
        {
            var (exitCode, stdout, _) = Run("260", dir, "--scratch-dir", ScratchDirectory);

            Assert.Equal(0, exitCode);
            Assert.Contains("Training BPE tokeniser on 1 file(s)", stdout);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
            File.Delete(vocabPath);
        }
    }

    [Fact]
    public void IsTmpfs_KnownTmpfsPath_ReturnsTrue()
    {
        Assert.True(TokeniserCli.IsTmpfs("/tmp"));
    }

    [Fact]
    public void IsTmpfs_KnownRealDiskPath_ReturnsFalse()
    {
        Assert.False(TokeniserCli.IsTmpfs("/home"));
    }

    [Fact]
    public void EstimateScratchBytes_IsFourInt32sPerInputByte()
    {
        Assert.Equal(1000L * 16, TokeniserCli.EstimateScratchBytes(1000));
    }

    [Fact]
    public void ExceedsDiskBudget_WithinEightyPercentOfAvailable_ReturnsFalse()
    {
        Assert.False(TokeniserCli.ExceedsDiskBudget(estimatedScratchBytes: 79, availableDiskBytes: 100));
    }

    [Fact]
    public void ExceedsDiskBudget_AboveEightyPercentOfAvailable_ReturnsTrue()
    {
        Assert.True(TokeniserCli.ExceedsDiskBudget(estimatedScratchBytes: 81, availableDiskBytes: 100));
    }
}
