using System;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.NegativeSampler
{
    public class CDNegativeSamplerFactory : INegativeSamplerFactory
    {
        public INegativeSampler CreateNegativeSampler(
            ICalculator calculator, 
            float[] weights, 
            float[] visible, 
            float[] hidden0, 
            float[] hidden1)
        {
            if (calculator == null)
            {
                throw new ArgumentNullException("calculator");
            }
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visible == null)
            {
                throw new ArgumentNullException("visible");
            }
            if (hidden0 == null)
            {
                throw new ArgumentNullException("hidden0");
            }
            if (hidden1 == null)
            {
                throw new ArgumentNullException("hidden1");
            }

            return 
                new CD(
                    calculator,
                    weights,
                    visible,
                    hidden0,
                    hidden1);
        }
    }
}