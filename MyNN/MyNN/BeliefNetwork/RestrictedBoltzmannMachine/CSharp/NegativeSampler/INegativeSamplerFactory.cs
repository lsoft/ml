using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.NegativeSampler
{
    public interface INegativeSamplerFactory
    {
        INegativeSampler CreateNegativeSampler(
            ICalculator calculator,
            float[] weights,
            float[] visible,
            float[] hidden0,
            float[] hidden1
            );
    }
}