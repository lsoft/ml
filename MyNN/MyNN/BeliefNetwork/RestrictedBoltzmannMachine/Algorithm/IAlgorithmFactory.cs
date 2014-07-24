using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm
{
    public interface IAlgorithmFactory<T>
        where T : IContainer
    {
        IAlgorithm CreateAlgorithm(
            ICalculator calculator,
            T container
            );
    }
}