using System;
using MyNN.Common.Randomizer;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.Calculator.CSharp
{
    public class NaiveConvolutionCalculator : ICSharpConvolutionCalculator
    {
        public void CalculateBackConvolutionWithIncrement(
            IReferencedSquareFloat kernelBiasContainer, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dataToConvolute, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            )
        {
            this.CalculateBackConvolution(
                kernelBiasContainer,
                dataToConvolute,
                target,
                false
                );
        }


        public void CalculateBackConvolutionWithOverwrite(
            IReferencedSquareFloat kernelBiasContainer, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dataToConvolute, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            )
        {
            this.CalculateBackConvolution(
                kernelBiasContainer,
                dataToConvolute,
                target,
                true
                );
        }

        public void CalculateConvolutionWithIncrement(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target
            )
        {
            this.CalculateConvolution(
                kernelBiasContainer,
                dataToConvolute,
                target,
                false
                );
        }

        public void CalculateConvolutionWithOverwrite(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target
            )
        {
            this.CalculateConvolution(
                kernelBiasContainer,
                dataToConvolute,
                target,
                true
                );
        }

        #region private code

        private void CalculateConvolution(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target,
            bool overwrite
            )
        {
            if (kernelBiasContainer == null)
            {
                throw new ArgumentNullException("kernelBiasContainer");
            }
            if (dataToConvolute == null)
            {
                throw new ArgumentNullException("dataToConvolute");
            }
            if (target == null)
            {
                throw new ArgumentNullException("target");
            }

            //делаем свертку
            for (var i = 0; i < target.Width; i++)
            {
                for (var j = 0; j < target.Height; j++)
                {
                    var zSum = 0f;
                    for (var a = 0; a < kernelBiasContainer.Width; a++)
                    {
                        for (var b = 0; b < kernelBiasContainer.Height; b++)
                        {
                            var w = kernelBiasContainer.GetValueFromCoordSafely(a, b);
                            var y = dataToConvolute.GetValueFromCoordSafely(i + a, j + b);

                            var z = w * y;
                            zSum += z;
                        }
                    }

                    zSum += kernelBiasContainer.Bias;

                    if (overwrite)
                    {
                        target.SetValueFromCoordSafely(
                            i,
                            j,
                            zSum
                            );
                    }
                    else
                    {
                        target.AddValueFromCoordSafely(
                            i,
                            j,
                            zSum
                            );
                    }
                }
            }
        }

        private void CalculateBackConvolution(
            IReferencedSquareFloat kernelBiasContainer, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dataToConvolute, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target, //think as current layer (e.g. pooling layer) dE\dy
            bool overwrite
            )
        {
            if (kernelBiasContainer == null)
            {
                throw new ArgumentNullException("kernelBiasContainer");
            }
            if (dataToConvolute == null)
            {
                throw new ArgumentNullException("dataToConvolute");
            }
            if (target == null)
            {
                throw new ArgumentNullException("target");
            }

            //делаем свертку
            for (var i = 0; i < target.Width; i++)
            {
                for (var j = 0; j < target.Height; j++)
                {
                    var zSum = 0f;
                    for (var a = 0; a < kernelBiasContainer.Width; a++)
                    {
                        for (var b = 0; b < kernelBiasContainer.Height; b++)
                        {
                            var w = kernelBiasContainer.GetValueFromCoordSafely(a, b);
                            var y = dataToConvolute.GetValueFromCoordPaddedWithZeroes(i - a, j - b);

                            var z = w * y;
                            zSum += z;
                        }
                    }

                    if (overwrite)
                    {
                        target.SetValueFromCoordSafely(
                            i,
                            j,
                            zSum
                            );
                    }
                    else
                    {
                        target.AddValueFromCoordSafely(
                            i,
                            j,
                            zSum
                            );
                    }
                }
            }
        }

        #endregion

    }
}