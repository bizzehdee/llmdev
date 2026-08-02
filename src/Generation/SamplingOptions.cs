namespace Generation;

/// <summary>
/// How to turn a model's next-token logits into an actual chosen token.
/// Temperature, top-k, and top-p can all be combined (top-k is applied
/// first, then top-p, both before the final softmax + sample); temperature
/// &lt;= 0 means plain greedy argmax and ignores top-k/top-p entirely, since
/// greedy is inherently a single deterministic choice.
/// </summary>
public sealed record SamplingOptions
{
    /// <summary>&lt;= 0 means greedy. 1 is unmodified. &lt;1 sharpens (more deterministic), &gt;1 flattens (more random).</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>If set, only the K highest-logit tokens are eligible.</summary>
    public int? TopK { get; init; }

    /// <summary>If set, only the smallest set of highest-probability tokens whose cumulative probability reaches this threshold are eligible ("nucleus sampling").</summary>
    public float? TopP { get; init; }

    public static SamplingOptions Greedy() => new() { Temperature = 0f };
}
