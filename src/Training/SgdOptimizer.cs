using Tensor;

namespace Training;

/// <summary>Plain stochastic gradient descent: w -= learningRate * grad.</summary>
public sealed class SgdOptimizer : IOptimizer
{
    private readonly IReadOnlyList<Variable> _parameters;
    private readonly float _learningRate;

    public SgdOptimizer(IReadOnlyList<Variable> parameters, float learningRate)
    {
        _parameters = parameters;
        _learningRate = learningRate;
    }

    public void Step()
    {
        foreach (var parameter in _parameters)
        {
            parameter.Value.SubtractInPlace(parameter.Gradient.Scale(_learningRate));
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
