using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.Tests.MLP2.ActivationFunction
{
    internal class ActivationFunctionDerivativeTests
    {
        private const float DeltaX = 0.01f;
        private const float DerivativeEpsilon = 0.001f;

        public void ExecuteTests(IFunction f)
        {
            if (f == null)
            {
                throw new ArgumentNullException("f");
            }

            for (var cc = -100.5f; cc < 100f; cc += 0.17f)
            {
                var center = f.Compute(cc);
                var left = f.Compute(cc - DeltaX);
                var right = f.Compute(cc + DeltaX);

                var cDerivative = (right - left)/(2f*DeltaX);
                var fDerivative = f.ComputeFirstDerivative(center);

                var diff = Math.Abs(cDerivative - fDerivative);

                if (diff >= DerivativeEpsilon)
                {
                    throw new Exception();
                }

                Assert.IsTrue(diff <= DerivativeEpsilon);
            }
        }
    }
}