using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2.ActivationFunction.CSharp
{
    internal class ActivationFunctionDerivativeTests
    {
        private const float DeltaX = 0.01f;

        public void ExecuteTests(
            IFunction f,
            float left = -100.5f,
            float right = 100f,
            float step = 0.17f,
            float allowedDerivativeEpsilon = 0.001f
            )
        {
            if (f == null)
            {
                throw new ArgumentNullException("f");
            }

            for (var cc = left; cc < right; cc += step)
            {
                var leftValue = f.Compute(cc - DeltaX);
                var rightValue = f.Compute(cc + DeltaX);

                var cDerivative = (rightValue - leftValue)/(2f*DeltaX);
                var fDerivative = f.ComputeFirstDerivative(cc);

                var diff = Math.Abs(cDerivative - fDerivative);

                Assert.IsTrue(diff <= allowedDerivativeEpsilon);
            }
        }
    }
}