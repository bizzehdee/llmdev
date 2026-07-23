using Tokeniser;

if (args.Length == 0)
{
    Console.WriteLine("Usage: Tokeniser <vocab-size> <file1.txt> [file2.txt ...]");
    Console.WriteLine("Trains a byte-level BPE vocabulary from the given text file(s) and");
    Console.WriteLine("demonstrates encoding/decoding a sample of the first file.");
    return 1;
}

if (!int.TryParse(args[0], out int vocabSize))
{
    Console.Error.WriteLine($"Expected <vocab-size> as the first argument, got '{args[0]}'.");
    return 1;
}

var files = args.Skip(1).ToArray();
if (files.Length == 0)
{
    Console.Error.WriteLine("Provide at least one input text file.");
    return 1;
}

foreach (var file in files)
{
    if (!File.Exists(file))
    {
        Console.Error.WriteLine($"File not found: {file}");
        return 1;
    }
}

Console.WriteLine($"Training BPE tokeniser on {files.Length} file(s), target vocab size {vocabSize}...");

var tokeniser = new BpeTokeniser();
tokeniser.Train(files, vocabSize);

Console.WriteLine($"Trained vocabulary size: {tokeniser.VocabSize}");

var sampleText = File.ReadAllText(files[0]);
if (sampleText.Length > 200)
{
    sampleText = sampleText[..200];
}

var encoded = tokeniser.Encode(sampleText);
var decoded = tokeniser.Decode(encoded);

Console.WriteLine();
Console.WriteLine("Sample text:");
Console.WriteLine(sampleText);
Console.WriteLine();
Console.WriteLine($"Encoded ({encoded.Count} tokens):");
Console.WriteLine(string.Join(" ", encoded));
Console.WriteLine();
Console.WriteLine("Decoded (should match sample text):");
Console.WriteLine(decoded);
Console.WriteLine();
Console.WriteLine($"Roundtrip match: {decoded == sampleText}");

const string vocabPath = "vocab.bpe";
tokeniser.Save(vocabPath);
Console.WriteLine();
Console.WriteLine($"Saved vocabulary + merges to {vocabPath}");

return 0;
