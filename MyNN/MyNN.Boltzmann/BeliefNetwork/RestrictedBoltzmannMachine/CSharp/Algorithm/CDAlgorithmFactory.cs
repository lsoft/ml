using System;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm
{
    public class CDAlgorithmFactory : IAlgorithmFactory<FloatArrayContainer>
    {
        public IAlgorithm CreateAlgorithm(
            ICalculator calculator,
            FloatArrayContainer container
            )
        {
            if (calculator == null)
            {
                throw new ArgumentNullException("calculator");
            }
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }


            return 
                new CD(
                    calculator,
                    container
                    );
        }
    }
}