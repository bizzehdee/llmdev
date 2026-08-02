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
    /// <remarks>
    /// Rather than rescanning the whole corpus on every merge (O(corpus size)
    /// per merge, hopeless past a few MB), tokens are held in a doubly linked
    /// list per source file (see <see cref="LinkedTokenStream"/>) and pair
    /// counts are updated incrementally: a merge only touches the positions
    /// where that exact pair occurs plus their immediate neighbours. A
    /// priority queue tracks the current best pair; stale entries (counts
    /// that changed after being enqueued) are detected on pop by comparing
    /// against the live count and discarded.
    ///
    /// Two things keep this from blowing up on a large corpus:
    ///
    /// 1. Memory for the corpus itself lives in memory-mapped scratch files
    ///    (<see cref="MappedInt32Array"/>), not the managed heap. The OS can
    ///    reclaim those pages directly (drop clean ones, write back dirty
    ///    ones) instead of only being able to page a plain array out to
    ///    swap - which is what exhausted this machine's RAM the first time
    ///    round.
    /// 2. Rather than a `Dictionary<pair, List<int>>` of occurrence
    ///    positions (a separate heap-allocated list per distinct pair, the
    ///    original source of the memory blowup), each pair's occurrences
    ///    are threaded through an *intrusive* singly linked list embedded in
    ///    a second mapped array (<c>PairNext</c>): `pairHead[pair]` is the
    ///    most recent position registered for that pair, and
    ///    `PairNext[position]` chains to the previous one. No collection
    ///    object per pair, so the only heap memory left is `pairHead` /
    ///    `pairCounts` / the priority queue, all bounded by the number of
    ///    *distinct* pairs, not the number of occurrences.
    /// </remarks>
    public void Train(IEnumerable<string> filePaths, int targetVocabSize, string scratchDirectory, Action<int, int>? onMerge = null)
    {
        if (targetVocabSize < 256)
        {
            throw new ArgumentOutOfRangeException(nameof(targetVocabSize), "Target vocab size must be at least 256 (the base byte vocabulary).");
        }

        using var state = LinkedTokenStream.Build(filePaths, scratchDirectory);
        var pairCounts = new Dictionary<(int Left, int Right), long>();
        var pairHead = new Dictionary<(int Left, int Right), int>();
        var heap = new PriorityQueue<(int Left, int Right), (long NegCount, int Left, int Right)>();

        // Pairs touched during the round in progress, flushed to the heap
        // once each at the end of the round rather than on every individual
        // count change. A common pair (e.g. a frequent English digraph) can
        // have millions of occurrences; pushing one heap entry per
        // occurrence rather than per *distinct* pair is what actually
        // exhausted RAM the first time this ran against a real corpus
        // (confirmed via dmesg: OOM-killed at ~6GB of heap entries). The
        // pairs actually touched by a round are bounded by current vocab
        // size (every new pair involves the just-created token id), so
        // deferring to a per-pair flush keeps this to a few thousand
        // entries per round regardless of how many occurrences it processed.
        var dirtyPairs = new HashSet<(int Left, int Right)>();

        void LinkOccurrence(int leftIndex, (int Left, int Right) pair)
        {
            int previousHead = pairHead.TryGetValue(pair, out int h) ? h : -1;
            state.PairNext[leftIndex] = previousHead;
            pairHead[pair] = leftIndex;
        }

        void AdjustCount((int Left, int Right) pair, long delta)
        {
            pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + delta;
            dirtyPairs.Add(pair);
        }

        void FlushDirtyPairsToHeap()
        {
            foreach (var pair in dirtyPairs)
            {
                long count = pairCounts.GetValueOrDefault(pair);
                heap.Enqueue(pair, (-count, pair.Left, pair.Right));
            }
            dirtyPairs.Clear();
        }

        void AddPairAt(int leftIndex)
        {
            int rightIndex = state.Next[leftIndex];
            if (rightIndex == -1)
            {
                return;
            }

            var pair = (state.Token[leftIndex], state.Token[rightIndex]);
            LinkOccurrence(leftIndex, pair);
            AdjustCount(pair, 1);
        }

        // Initial scan: only build counts + occurrence chains here. Heap
        // population happens afterwards, once per distinct pair, instead of
        // once per occurrence (which for a large corpus is the difference
        // between a few hundred thousand heap entries and tens of millions).
        for (int i = 0; i < state.Length; i++)
        {
            int j = state.Next[i];
            if (j == -1)
            {
                continue;
            }

            var pair = (state.Token[i], state.Token[j]);
            LinkOccurrence(i, pair);
            pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + 1;
        }

        foreach (var (pair, count) in pairCounts)
        {
            heap.Enqueue(pair, (-count, pair.Left, pair.Right));
        }

        int nextId = 256;

        while (_vocab.Count < targetVocabSize)
        {
            (int Left, int Right) pair;
            long actualCount;
            while (true)
            {
                if (!heap.TryDequeue(out pair, out var priority))
                {
                    actualCount = 0;
                    break;
                }

                actualCount = pairCounts.GetValueOrDefault(pair);
                if (actualCount == -priority.NegCount && actualCount > 0)
                {
                    break;
                }
            }

            if (actualCount < 2)
            {
                break;
            }

            int newId = nextId++;
            _vocab[newId] = Combine(_vocab[pair.Left], _vocab[pair.Right]);
            _mergeRank[pair] = _merges.Count;
            _merges.Add((pair.Left, pair.Right, newId));
            onMerge?.Invoke(_vocab.Count, targetVocabSize);

            // Collect the full occurrence chain before merging anything.
            // Processing a node can overwrite *other* nodes' PairNext slot
            // (any position can only ever be mid-chain for one pair at a
            // time, and a merge changes what pair a neighbouring position
            // represents) - walking and mutating in the same pass would risk
            // severing the chain before every original occurrence is
            // reached. Collecting first also naturally excludes occurrences
            // *created* by this same merge round (e.g. "aaaa" -> merging
            // (a,a) creates a new (a,a) at the splice point): those get
            // linked onto a new head that this snapshot never sees, so
            // they're correctly left for a later round.
            var occurrences = new List<int>();
            int node = pairHead.TryGetValue(pair, out int headNode) ? headNode : -1;
            while (node != -1)
            {
                occurrences.Add(node);
                node = state.PairNext[node];
            }

            foreach (int i in occurrences)
            {
                if (state.Token[i] != pair.Left)
                {
                    continue;
                }

                int j = state.Next[i];
                if (j == -1 || state.Token[j] != pair.Right)
                {
                    continue;
                }

                int p = state.Prev[i];
                int k = state.Next[j];

                if (p != -1)
                {
                    AdjustCount((state.Token[p], state.Token[i]), -1);
                }
                AdjustCount(pair, -1);
                if (k != -1)
                {
                    AdjustCount((state.Token[j], state.Token[k]), -1);
                }

                state.Token[i] = newId;
                state.Token[j] = -1;
                state.Next[i] = k;
                if (k != -1)
                {
                    state.Prev[k] = i;
                }

                if (p != -1)
                {
                    AddPairAt(p);
                }
                AddPairAt(i);
            }

            FlushDirtyPairsToHeap();
        }
    }

    private static byte[] Combine(byte[] a, byte[] b)
    {
        var result = new byte[a.Length + b.Length];
        Buffer.BlockCopy(a, 0, result, 0, a.Length);
        Buffer.BlockCopy(b, 0, result, a.Length, b.Length);
        return result;
    }

    /// <summary>
    /// Holds every input document's bytes as one big doubly linked list
    /// (via index arrays) so merges can splice out a node in O(1) without
    /// shifting or copying the rest of the corpus. -1 means "no neighbour"
    /// (a tombstoned node, or the start/end of a document). <c>PairNext</c>
    /// is a second, separate linked structure threading together every
    /// position that currently represents the same adjacent pair (see the
    /// remarks on <see cref="Train"/>); it has nothing to do with document
    /// order. All four arrays are memory-mapped scratch files rather than
    /// managed arrays, since together they scale with corpus size and are
    /// the main thing that needs to stay off the process heap.
    /// </summary>
    private sealed class LinkedTokenStream : IDisposable
    {
        public required MappedInt32Array Token;
        public required MappedInt32Array Next;
        public required MappedInt32Array Prev;
        public required MappedInt32Array PairNext;
        public required int Length;

        public static LinkedTokenStream Build(IEnumerable<string> filePaths, string scratchDirectory)
        {
            var documents = filePaths
                .Select(File.ReadAllBytes)
                .Where(bytes => bytes.Length > 1)
                .ToList();

            int total = documents.Sum(d => d.Length);
            var token = new MappedInt32Array(total, scratchDirectory);
            var next = new MappedInt32Array(total, scratchDirectory);
            var prev = new MappedInt32Array(total, scratchDirectory);
            var pairNext = new MappedInt32Array(total, scratchDirectory);

            int offset = 0;
            foreach (var doc in documents)
            {
                for (int i = 0; i < doc.Length; i++)
                {
                    int index = offset + i;
                    token[index] = doc[i];
                    prev[index] = i == 0 ? -1 : index - 1;
                    next[index] = i == doc.Length - 1 ? -1 : index + 1;
                }
                offset += doc.Length;
            }

            return new LinkedTokenStream { Token = token, Next = next, Prev = prev, PairNext = pairNext, Length = total };
        }

        public void Dispose()
        {
            Token.Dispose();
            Next.Dispose();
            Prev.Dispose();
            PairNext.Dispose();
        }
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
