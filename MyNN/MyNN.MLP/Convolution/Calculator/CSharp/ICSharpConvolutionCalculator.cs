using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.Calculator.CSharp
{
    public interface ICSharpConvolutionCalculator
    {
        void CalculateBackConvolutionWithIncrement(
            IReferencedSquareFloat kernelBiasContainer, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dataToConvolute, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            );

        void CalculateBackConvolutionWithOverwrite(
            IReferencedSquareFloat kernelBiasContainer, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dataToConvolute, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            );

        void CalculateConvolutionWithOverwrite(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target
            );

        void CalculateConvolutionWithIncrement(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target
            );

    }
}