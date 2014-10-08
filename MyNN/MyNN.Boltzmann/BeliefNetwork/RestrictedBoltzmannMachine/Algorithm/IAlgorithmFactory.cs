using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm
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