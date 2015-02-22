using System;
using System.Security.Cryptography;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.ErrorCalculator.CSharp
{
    public class NaiveErrorCalculator : IErrorCalculator
    {
        public void CalculateError(
            IReferencedSquareFloat net,
            IReferencedSquareFloat state,
            float[] desiredValues,
            IMetrics e,
            IFunction activationFunction,
            IReferencedSquareFloat dedz
            )
        {
            if (net == null)
            {
                throw new ArgumentNullException("net");
            }
            if (state == null)
            {
                throw new ArgumentNullException("state");
            }
            if (desiredValues == null)
            {
                throw new ArgumentNullException("desiredValues");
            }
            if (e == null)
            {
                throw new ArgumentNullException("e");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }
            if (dedz == null)
            {
                throw new ArgumentNullException("dedz");
            }

            var pseudoState = new float[desiredValues.Length];

            for (var h = 0; h < state.Height; h++)
            {
                for (var w = 0; w < state.Width; w++)
                {
                    var index = h*state.Width + w;

                    pseudoState[index] = state.GetValueFromCoordSafely(w, h);
                }
            }

            for (var h = 0; h < state.Height; h++)
            {
                for (var w = 0; w < state.Width; w++)
                {
                    var index = h * state.Width + w;

                    var dedy = e.CalculatePartialDerivativeByV2Index(
                        pseudoState,
                        desiredValues,
                        index
                        );

                    var z = net.GetValueFromCoordSafely(w, h);
                    var sigma_sh = activationFunction.ComputeFirstDerivative(z);

                    var dEdz = dedy * sigma_sh;

                    dedz.SetValueFromCoordSafely(w, h, dEdz);
                }
            }
        }
    }
}