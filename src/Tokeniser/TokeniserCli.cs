namespace Tokeniser;

/// <summary>
/// The tokeniser CLI's actual logic (argument parsing/validation, tmpfs
/// detection, disk-space checking, training + a sample encode/decode
/// roundtrip), factored out of <c>Program.cs</c> so it can be unit tested
/// directly - top-level statements can't be invoked from a test project.
/// Takes its output streams as parameters (rather than hardcoding
/// <see cref="Console"/>) so tests can capture and assert on CLI output
/// without touching the real console.
/// </summary>
public static class TokeniserCli
{
    // The corpus itself lives in memory-mapped scratch files (see
    // MappedArray<int> / LinkedTokenStream) rather than the managed heap, so
    // training doesn't need a RAM budget check - the OS can reclaim those
    // pages under pressure instead of them threatening to OOM the machine.
    // The one real resource to guard is disk space for that scratch: 4
    // int32 arrays (Token, Next, Prev, PairNext), one entry per byte of input.
    private const int BytesPerScratchEntry = 4 * sizeof(int);
    private const double MaxFractionOfAvailableDisk = 0.8;

    public static int Run(string[] args, TextWriter stdout, TextWriter stderr)
    {
        if (args.Length == 0)
        {
            stdout.WriteLine("Usage: Tokeniser <vocab-size> <file-or-directory> [file-or-directory ...] [--scratch-dir <dir>]");
            stdout.WriteLine("Trains a byte-level BPE vocabulary from the given text file(s) and");
            stdout.WriteLine("demonstrates encoding/decoding a sample of the first file.");
            stdout.WriteLine("Directories are expanded to their *.txt files (non-.txt files, e.g. .epub, are skipped).");
            stdout.WriteLine($"Training scratch data is memory-mapped to disk (~{BytesPerScratchEntry} bytes per byte of");
            stdout.WriteLine("input), so it won't exhaust system RAM; before starting, available disk space is checked instead.");
            stdout.WriteLine("The scratch directory must be real disk, not a tmpfs mount (e.g. /tmp on many Linux distros) -");
            stdout.WriteLine("that would just be RAM under a different name. Defaults to ~/.local/share/tokeniser-scratch.");
            return 1;
        }

        if (!int.TryParse(args[0], out int vocabSize))
        {
            stderr.WriteLine($"Expected <vocab-size> as the first argument, got '{args[0]}'.");
            return 1;
        }

        var positionalArgs = new List<string>();
        string scratchDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "tokeniser-scratch");
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--scratch-dir")
            {
                if (i + 1 >= args.Length)
                {
                    stderr.WriteLine("--scratch-dir requires a path argument.");
                    return 1;
                }
                scratchDirectory = args[i + 1];
                i++;
            }
            else
            {
                positionalArgs.Add(args[i]);
            }
        }

        var inputs = positionalArgs.ToArray();
        if (inputs.Length == 0)
        {
            stderr.WriteLine("Provide at least one input text file or directory.");
            return 1;
        }

        Directory.CreateDirectory(scratchDirectory);
        if (IsTmpfs(scratchDirectory))
        {
            stderr.WriteLine($"Refusing to train: scratch directory {scratchDirectory} is on a tmpfs mount, which is");
            stderr.WriteLine("RAM, not disk - using it would recreate exactly the problem this scratch space exists to avoid.");
            stderr.WriteLine("Pick a directory on real disk with --scratch-dir <dir>.");
            return 1;
        }

        var files = new List<string>();
        foreach (var input in inputs)
        {
            if (Directory.Exists(input))
            {
                files.AddRange(Directory.GetFiles(input, "*.txt"));
            }
            else if (File.Exists(input))
            {
                files.Add(input);
            }
            else
            {
                stderr.WriteLine($"File or directory not found: {input}");
                return 1;
            }
        }

        if (files.Count == 0)
        {
            stderr.WriteLine("No .txt files found in the given input(s).");
            return 1;
        }

        long totalBytes = files.Sum(f => new FileInfo(f).Length);
        long estimatedScratchBytes = EstimateScratchBytes(totalBytes);
        long availableDiskBytes = new DriveInfo(scratchDirectory).AvailableFreeSpace;
        if (ExceedsDiskBudget(estimatedScratchBytes, availableDiskBytes))
        {
            stderr.WriteLine(
                $"Refusing to train: {totalBytes / 1024.0 / 1024.0:F1} MB of input needs an estimated " +
                $"~{estimatedScratchBytes / 1024.0 / 1024.0:F0} MB of disk scratch space in {scratchDirectory}, " +
                $"which exceeds {MaxFractionOfAvailableDisk:P0} of the {availableDiskBytes / 1024.0 / 1024.0:F0} MB free there.");
            stderr.WriteLine("Free up disk space, pass a different --scratch-dir, or use fewer input files and retry.");
            return 1;
        }

        stdout.WriteLine($"Training BPE tokeniser on {files.Count} file(s), {totalBytes / 1024.0 / 1024.0:F1} MB, " +
            $"target vocab size {vocabSize} (~{estimatedScratchBytes / 1024.0 / 1024.0:F0} MB disk scratch in {scratchDirectory})...");

        var tokeniser = new BpeTokeniser();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        tokeniser.Train(files, vocabSize, scratchDirectory, onMerge: (currentVocabSize, targetVocabSize) =>
        {
            if (currentVocabSize % 100 == 0 || currentVocabSize == targetVocabSize)
            {
                stdout.WriteLine($"  vocab {currentVocabSize}/{targetVocabSize} ({stopwatch.Elapsed:mm\\:ss})");
            }
        });

        stdout.WriteLine($"Trained vocabulary size: {tokeniser.VocabSize} in {stopwatch.Elapsed:mm\\:ss}");

        var sampleText = File.ReadAllText(files[0]);
        if (sampleText.Length > 200)
        {
            sampleText = sampleText[..200];
        }

        var encoded = tokeniser.Encode(sampleText);
        var decoded = tokeniser.Decode(encoded);

        stdout.WriteLine();
        stdout.WriteLine("Sample text:");
        stdout.WriteLine(sampleText);
        stdout.WriteLine();
        stdout.WriteLine($"Encoded ({encoded.Count} tokens):");
        stdout.WriteLine(string.Join(" ", encoded));
        stdout.WriteLine();
        stdout.WriteLine("Decoded (should match sample text):");
        stdout.WriteLine(decoded);
        stdout.WriteLine();
        stdout.WriteLine($"Roundtrip match: {decoded == sampleText}");

        const string vocabPath = "vocab.bpe";
        tokeniser.Save(vocabPath);
        stdout.WriteLine();
        stdout.WriteLine($"Saved vocabulary + merges to {vocabPath}");

        return 0;
    }

    /// <summary>Estimated disk scratch usage for a corpus of this size - see BytesPerScratchEntry's comment.</summary>
    public static long EstimateScratchBytes(long totalInputBytes) => totalInputBytes * BytesPerScratchEntry;

    /// <summary>Whether training would use more than <see cref="MaxFractionOfAvailableDisk"/> of the available disk space.</summary>
    public static bool ExceedsDiskBudget(long estimatedScratchBytes, long availableDiskBytes) =>
        estimatedScratchBytes > (long)(availableDiskBytes * MaxFractionOfAvailableDisk);

    public static bool IsTmpfs(string directory)
    {
        if (!OperatingSystem.IsLinux())
        {
            return false;
        }

        try
        {
            string resolved = Path.GetFullPath(directory);
            string? bestMountPoint = null;
            string? bestFsType = null;

            foreach (var line in File.ReadLines("/proc/mounts"))
            {
                var fields = line.Split(' ');
                if (fields.Length < 3)
                {
                    continue;
                }

                string mountPoint = fields[1];
                if ((resolved == mountPoint || resolved.StartsWith(mountPoint.TrimEnd('/') + "/", StringComparison.Ordinal) || mountPoint == "/")
                    && (bestMountPoint is null || mountPoint.Length > bestMountPoint.Length))
                {
                    bestMountPoint = mountPoint;
                    bestFsType = fields[2];
                }
            }

            return bestFsType == "tmpfs";
        }
        catch (IOException)
        {
            return false;
        }
    }
}
