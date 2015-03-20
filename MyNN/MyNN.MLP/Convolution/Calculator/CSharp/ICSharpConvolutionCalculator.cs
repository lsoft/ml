using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.Calculator.CSharp
{
    /// <summary>
    /// Выполнение свертки
    /// </summary>
    public interface ICSharpConvolutionCalculator
    {
        /// <summary>
        /// Выполнение свертки в "обратном" направлении. То есть, с учетом
        /// нулей, лежащих за границами. Значения светки добавляются в target.
        /// </summary>
        void CalculateBackConvolutionWithIncrement(
            IReferencedSquareFloat dzdy, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dedz, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            );

        /// <summary>
        /// Выполнение свертки в "обратном" направлении. То есть, с учетом
        /// нулей, лежащих за границами. Значения светки перезаписывают target.
        /// </summary>
        void CalculateBackConvolutionWithOverwrite(
            IReferencedSquareFloat dzdy, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dedz, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            );

        /// <summary>
        /// Выполнение свертки в "прямом" направлении. То есть, без учета
        /// нулей, лежащих за границами. Значения светки перезаписывают target.
        /// </summary>
        void CalculateConvolutionWithOverwrite(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target
            );

        /// <summary>
        /// Выполнение свертки в "прямом" направлении. То есть, без учета
        /// нулей, лежащих за границами. Значения светки добавляются в target.
        /// </summary>
        void CalculateConvolutionWithIncrement(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target
            );

    }
}