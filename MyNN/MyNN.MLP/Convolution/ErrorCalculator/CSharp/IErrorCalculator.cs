using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.ErrorCalculator.CSharp
{
    public interface IErrorCalculator
    {
        void CalculateError(
            IReferencedSquareFloat net,
            IReferencedSquareFloat state,
            float[] desiredValues,
            IMetrics e,
            IFunction activationFunction,
            IReferencedSquareFloat dedz
            );
    }
}
