using System;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm
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