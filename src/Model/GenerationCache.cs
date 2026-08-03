using TensorValue = Tensor.Tensor;

namespace Model;

/// <summary>
/// Holds per-layer cached Key/Value tensors for KV-cached autoregressive
/// generation (TASK-020): <see cref="GptModel.ForwardIncremental"/> only
/// computes Q/K/V for the *new* tokens each step, then grows this cache by
/// concatenating those new K/V onto whatever's already cached - so
/// attention still sees every position's keys/values without ever
/// recomputing the ones already processed. Plain <see cref="TensorValue"/>,
/// not <see cref="Tensor.Variable"/>: generation never backpropagates, so
/// there's nothing for the cache itself to need an autodiff graph for.
/// </summary>
public sealed class GenerationCache : IDisposable
{
    private readonly TensorValue?[] _keys;
    private readonly TensorValue?[] _values;

    /// <summary>How many positions have been cached so far (0 before the first call).</summary>
    public int Length { get; private set; }

    public GenerationCache(int numLayers)
    {
        _keys = new TensorValue?[numLayers];
        _values = new TensorValue?[numLayers];
    }

    internal TensorValue? GetKey(int layer) => _keys[layer];
    internal TensorValue? GetValue(int layer) => _values[layer];

    internal void SetLayer(int layer, TensorValue key, TensorValue value)
    {
        _keys[layer]?.Dispose();
        _values[layer]?.Dispose();
        _keys[layer] = key;
        _values[layer] = value;
    }

    internal void AdvanceLength(int by) => Length += by;

    /// <summary>
    /// Discards every layer's cached state, back to empty. Used when a
    /// sliding-context window truncates older tokens: KV-cache positions
    /// can't simply be shifted, so the cache is rebuilt from scratch for
    /// the truncated window instead (see <see cref="Generation.TextGenerator"/>).
    /// </summary>
    public void Reset()
    {
        for (int layer = 0; layer < _keys.Length; layer++)
        {
            _keys[layer]?.Dispose();
            _values[layer]?.Dispose();
            _keys[layer] = null;
            _values[layer] = null;
        }
        Length = 0;
    }

    public void Dispose()
    {
        foreach (var key in _keys)
        {
            key?.Dispose();
        }
        foreach (var value in _values)
        {
            value?.Dispose();
        }
    }
}
