using Tensor;
using TensorValue = Tensor.Tensor;

namespace Training;

/// <summary>
/// AdamW: Adam's per-parameter adaptive learning rate (via running first
/// and second moment estimates of the gradient), plus weight decay applied
/// directly to the weights rather than folded into the gradient (the "W" -
/// decoupled weight decay - Adam with L2 regularisation baked into the
/// gradient instead behaves subtly differently once you're also adapting
/// the learning rate per-parameter, which is the whole reason AdamW exists).
///
/// Per-parameter moment estimates default to plain heap tensors, same size
/// as the parameter itself (so this optimizer's own state is ~2x the
/// model's parameter count). TASK-019: pass <c>useDiskBackedState</c> to
/// back them with <see cref="TensorValue.ZerosOnDisk"/> instead, mirroring
/// <c>MappedArray&lt;T&gt;</c>'s scratch-file pattern - worth it once a
/// model's parameter count makes 2x its size in RAM matter, not by
/// default (nothing built in this project so far has been that large).
/// Deliberately lower priority than TASK-017/018/020: cheap to build when
/// the time comes, not worth speculatively building ahead of an actual
/// need.
/// </summary>
public sealed class AdamWOptimizer : IOptimizer, IDisposable
{
    private readonly IReadOnlyList<Variable> _parameters;
    private readonly float _learningRate;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    private readonly bool _useDiskBackedState;
    private readonly Dictionary<Variable, TensorValue> _firstMoment = new();
    private readonly Dictionary<Variable, TensorValue> _secondMoment = new();
    private int _step;

    public AdamWOptimizer(IReadOnlyList<Variable> parameters, float learningRate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0.01f, bool useDiskBackedState = false, string? scratchDirectory = null)
    {
        if (useDiskBackedState && string.IsNullOrEmpty(scratchDirectory))
        {
            throw new ArgumentException("scratchDirectory is required when useDiskBackedState is true.", nameof(scratchDirectory));
        }

        _parameters = parameters;
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _useDiskBackedState = useDiskBackedState;

        foreach (var parameter in parameters)
        {
            _firstMoment[parameter] = useDiskBackedState ? TensorValue.ZerosOnDisk(parameter.Value.Shape, scratchDirectory!) : TensorValue.Zeros(parameter.Value.Shape);
            _secondMoment[parameter] = useDiskBackedState ? TensorValue.ZerosOnDisk(parameter.Value.Shape, scratchDirectory!) : TensorValue.Zeros(parameter.Value.Shape);
        }
    }

    public void Step()
    {
        _step++;
        float biasCorrection1 = 1f - MathF.Pow(_beta1, _step);
        float biasCorrection2 = 1f - MathF.Pow(_beta2, _step);
        var epsilonTensor = TensorValue.FromValues([_epsilon], [1]);

        foreach (var parameter in _parameters)
        {
            var gradient = parameter.Gradient;

            var mComputed = _firstMoment[parameter].Scale(_beta1).Add(gradient.Scale(1f - _beta1));
            var vComputed = _secondMoment[parameter].Scale(_beta2).Add(gradient.Multiply(gradient).Scale(1f - _beta2));
            UpdateMoment(_firstMoment, parameter, mComputed);
            UpdateMoment(_secondMoment, parameter, vComputed);

            var mHat = _firstMoment[parameter].Scale(1f / biasCorrection1);
            var vHat = _secondMoment[parameter].Scale(1f / biasCorrection2);

            var adaptiveStep = mHat.Divide(vHat.Sqrt().Add(epsilonTensor)).Scale(_learningRate);
            var decayStep = parameter.Value.Scale(_learningRate * _weightDecay);

            parameter.Value.SubtractInPlace(adaptiveStep.Add(decayStep));
        }
    }

    /// <summary>
    /// Heap-backed state (the default) simply swaps in the freshly computed
    /// moment tensor, same as before this task. Disk-backed state instead
    /// copies the computed values into the *existing* disk-backed tensor
    /// and disposes the transient heap-backed one - the persistent
    /// disk-backed object is never replaced, so its scratch file is opened
    /// once per parameter for the optimizer's whole lifetime rather than
    /// once per Step() call (which would otherwise leak a scratch file per
    /// parameter per step, since nothing else would ever Dispose the
    /// replaced one).
    /// </summary>
    private void UpdateMoment(Dictionary<Variable, TensorValue> moments, Variable parameter, TensorValue computed)
    {
        if (_useDiskBackedState)
        {
            moments[parameter].LoadInPlace(computed.ToArray());
            computed.Dispose();
        }
        else
        {
            moments[parameter] = computed;
        }
    }

    public void ZeroGrad()
    {
        foreach (var parameter in _parameters)
        {
            parameter.ZeroGrad();
        }
    }

    /// <summary>
    /// No-op for the default heap-backed state (heap tensors' Dispose is
    /// itself a no-op). Required for disk-backed state, to release the
    /// moment tensors' mapped scratch files - callers using
    /// <c>useDiskBackedState: true</c> must Dispose this optimizer once
    /// training finishes.
    /// </summary>
    public void Dispose()
    {
        foreach (var moment in _firstMoment.Values)
        {
            moment.Dispose();
        }
        foreach (var moment in _secondMoment.Values)
        {
            moment.Dispose();
        }
    }
}
