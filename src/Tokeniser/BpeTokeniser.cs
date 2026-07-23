using System.Text;

namespace Tokeniser;

/// <summary>
/// Byte-level Byte-Pair Encoding tokeniser, implemented from first principles
/// (no external tokenisation libraries). Base vocabulary is the 256 possible
/// byte values; training repeatedly merges the most frequent adjacent pair
/// of tokens until the target vocabulary size is reached.
/// </summary>
public sealed class BpeTokeniser
{
    private readonly Dictionary<int, byte[]> _vocab = new();
    private readonly Dictionary<(int Left, int Right), int> _mergeRank = new();
    private readonly List<(int Left, int Right, int NewId)> _merges = new();

    public int VocabSize => _vocab.Count;

    public BpeTokeniser()
    {
        for (int b = 0; b < 256; b++)
        {
            _vocab[b] = new[] { (byte)b };
        }
    }

    /// <summary>
    /// Trains merges from one or more input text files until either
    /// <paramref name="targetVocabSize"/> is reached or no pair occurs more
    /// than once.
    /// </summary>
    public void Train(IEnumerable<string> filePaths, int targetVocabSize)
    {
        if (targetVocabSize < 256)
        {
            throw new ArgumentOutOfRangeException(nameof(targetVocabSize), "Target vocab size must be at least 256 (the base byte vocabulary).");
        }

        var sequences = filePaths
            .Select(path => File.ReadAllBytes(path).Select(b => (int)b).ToList())
            .Where(seq => seq.Count > 1)
            .ToList();

        int nextId = 256;

        while (_vocab.Count < targetVocabSize)
        {
            var pairCounts = CountPairs(sequences);
            if (pairCounts.Count == 0)
            {
                break;
            }

            var best = pairCounts
                .OrderByDescending(kv => kv.Value)
                .ThenBy(kv => kv.Key.Left)
                .ThenBy(kv => kv.Key.Right)
                .First();

            if (best.Value < 2)
            {
                break;
            }

            var pair = best.Key;
            int newId = nextId++;

            _vocab[newId] = Combine(_vocab[pair.Left], _vocab[pair.Right]);
            _mergeRank[pair] = _merges.Count;
            _merges.Add((pair.Left, pair.Right, newId));

            foreach (var seq in sequences)
            {
                MergePairInPlace(seq, pair, newId);
            }
        }
    }

    private static byte[] Combine(byte[] a, byte[] b)
    {
        var result = new byte[a.Length + b.Length];
        Buffer.BlockCopy(a, 0, result, 0, a.Length);
        Buffer.BlockCopy(b, 0, result, a.Length, b.Length);
        return result;
    }

    private static Dictionary<(int Left, int Right), int> CountPairs(List<List<int>> sequences)
    {
        var counts = new Dictionary<(int Left, int Right), int>();
        foreach (var seq in sequences)
        {
            for (int i = 0; i < seq.Count - 1; i++)
            {
                var pair = (seq[i], seq[i + 1]);
                counts[pair] = counts.GetValueOrDefault(pair) + 1;
            }
        }
        return counts;
    }

    private static void MergePairInPlace(List<int> seq, (int Left, int Right) pair, int newId)
    {
        int write = 0;
        int read = 0;
        while (read < seq.Count)
        {
            if (read < seq.Count - 1 && seq[read] == pair.Left && seq[read + 1] == pair.Right)
            {
                seq[write++] = newId;
                read += 2;
            }
            else
            {
                seq[write++] = seq[read++];
            }
        }
        seq.RemoveRange(write, seq.Count - write);
    }

    /// <summary>
    /// Encodes text into token IDs by starting from raw UTF-8 bytes and
    /// repeatedly applying the learned merge with the lowest rank (i.e. the
    /// merge learned earliest during training) until no merge applies.
    /// </summary>
    public List<int> Encode(string text)
    {
        var ids = Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList();

        while (ids.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;

            for (int i = 0; i < ids.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((ids[i], ids[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1)
            {
                break;
            }

            var merge = _merges[bestRank];
            MergePairInPlace(ids, (merge.Left, merge.Right), merge.NewId);
        }

        return ids;
    }

    /// <summary>
    /// Decodes token IDs back into text by concatenating each token's raw
    /// bytes and decoding the resulting byte stream as UTF-8.
    /// </summary>
    public string Decode(IEnumerable<int> ids)
    {
        using var stream = new MemoryStream();
        foreach (int id in ids)
        {
            var bytes = _vocab[id];
            stream.Write(bytes, 0, bytes.Length);
        }
        return Encoding.UTF8.GetString(stream.ToArray());
    }

    public void Save(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine(_vocab.Count);
        foreach (var (id, bytes) in _vocab.OrderBy(kv => kv.Key))
        {
            writer.WriteLine($"{id} {Convert.ToBase64String(bytes)}");
        }

        writer.WriteLine(_merges.Count);
        foreach (var (left, right, newId) in _merges)
        {
            writer.WriteLine($"{left} {right} {newId}");
        }
    }

    public static BpeTokeniser Load(string path)
    {
        var tokeniser = new BpeTokeniser();
        using var reader = new StreamReader(path);

        int vocabCount = int.Parse(reader.ReadLine() ?? "0");
        tokeniser._vocab.Clear();
        for (int i = 0; i < vocabCount; i++)
        {
            var parts = (reader.ReadLine() ?? "").Split(' ', 2);
            int id = int.Parse(parts[0]);
            tokeniser._vocab[id] = Convert.FromBase64String(parts[1]);
        }

        int mergeCount = int.Parse(reader.ReadLine() ?? "0");
        for (int i = 0; i < mergeCount; i++)
        {
            var parts = (reader.ReadLine() ?? "").Split(' ');
            int left = int.Parse(parts[0]);
            int right = int.Parse(parts[1]);
            int newId = int.Parse(parts[2]);
            tokeniser._merges.Add((left, right, newId));
            tokeniser._mergeRank[(left, right)] = tokeniser._merges.Count - 1;
        }

        return tokeniser;
    }
}
