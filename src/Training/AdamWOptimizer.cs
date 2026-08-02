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
/// Per-parameter moment estimates are plain heap tensors for now, same
/// size as the parameter itself (so this optimizer's own state is ~2x the
/// model's parameter count) - TASK-012's job to decide whether that needs
/// disk-backing once model size makes it non-trivial (see PLAN.md/TASK.md).
/// </summary>
public sealed class AdamWOptimizer : IOptimizer
{
    private readonly IReadOnlyList<Variable> _parameters;
    private readonly float _learningRate;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    private readonly Dictionary<Variable, TensorValue> _firstMoment = new();
    private readonly Dictionary<Variable, TensorValue> _secondMoment = new();
    private int _step;

    public AdamWOptimizer(IReadOnlyList<Variable> parameters, float learningRate = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0.01f)
    {
        _parameters = parameters;
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;

        foreach (var parameter in parameters)
        {
            _firstMoment[parameter] = TensorValue.Zeros(parameter.Value.Shape);
            _secondMoment[parameter] = TensorValue.Zeros(parameter.Value.Shape);
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

            var m = _firstMoment[parameter].Scale(_beta1).Add(gradient.Scale(1f - _beta1));
            var v = _secondMoment[parameter].Scale(_beta2).Add(gradient.Multiply(gradient).Scale(1f - _beta2));
            _firstMoment[parameter] = m;
            _secondMoment[parameter] = v;

            var mHat = m.Scale(1f / biasCorrection1);
            var vHat = v.Scale(1f / biasCorrection2);

            var adaptiveStep = mHat.Divide(vHat.Sqrt().Add(epsilonTensor)).Scale(_learningRate);
            var decayStep = parameter.Value.Scale(_learningRate * _weightDecay);

            parameter.Value.SubtractInPlace(adaptiveStep.Add(decayStep));
        }
    }

    public void ZeroGrad()
    {
        foreach (var parameter in _parameters)
        {
            parameter.ZeroGrad();
        }
    }
}
