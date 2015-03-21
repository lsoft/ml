using System;
using System.Threading.Tasks;
using MyNN.Common.Randomizer;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.Calculator.CSharp
{
    public class NaiveConvolutionCalculator : ICSharpConvolutionCalculator
    {
        public void CalculateBackConvolutionWithIncrement(
            IReferencedSquareFloat dzdy, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dedz, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            )
        {
            this.CalculateBackConvolution(
                dzdy,
                dedz,
                target,
                1//false
                );
        }


        public void CalculateBackConvolutionWithOverwrite(
            IReferencedSquareFloat dzdy, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dedz, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target //think as current layer (e.g. pooling layer) dE\dy
            )
        {
            this.CalculateBackConvolution(
                dzdy,
                dedz,
                target,
                0//true
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
                1//false
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
                0//true
                );
        }

        #region private code

        private void CalculateConvolution(
            IReferencedKernelBiasContainer kernelBiasContainer,
            IReferencedSquareFloat dataToConvolute,
            IReferencedSquareFloat target,
            int invert_overwrite
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
            Parallel.For(0, target.Height, j =>
            //for (var j = 0; j < target.Height; j++)
            {
                for (var i = 0; i < target.Width; i++)
                {
                    var zSum = 0f;
                    for (var b = 0; b < kernelBiasContainer.Height; b++)
                    {
                        for (var a = 0; a < kernelBiasContainer.Width; a++)
                        {
                            var w = kernelBiasContainer.GetValueFromCoordSafely(a, b);
                            var y = dataToConvolute.GetValueFromCoordSafely(i + a, j + b);

                            var z = w * y;
                            zSum += z;
                        }
                    }

                    zSum += kernelBiasContainer.Bias;

                    target.ChangeValueFromCoordSafely(
                        i,
                        j,
                        zSum,
                        invert_overwrite
                        );

                    //if(invert_overwrite == 0)
                    ////if (overwrite)
                    //{
                    //    target.SetValueFromCoordSafely(
                    //        i,
                    //        j,
                    //        zSum
                    //        );
                    //}
                    //else
                    //{
                    //    target.AddValueFromCoordSafely(
                    //        i,
                    //        j,
                    //        zSum
                    //        );
                    //}
                }
            }
            );//Parallel.For
        }

        private void CalculateBackConvolution(
            IReferencedSquareFloat dzdy, //think as next layer (e.g. convolution layer) dz\dy
            IReferencedSquareFloat dedz, //think as next layer (e.g. convolution layer) dE\dz
            IReferencedSquareFloat target, //think as current layer (e.g. pooling layer) dE\dy
            int invert_overwrite
            )
        {
            if (dzdy == null)
            {
                throw new ArgumentNullException("dzdy");
            }
            if (dedz == null)
            {
                throw new ArgumentNullException("dedz");
            }
            if (target == null)
            {
                throw new ArgumentNullException("target");
            }

            //делаем свертку
            Parallel.For(0, target.Height, j =>
            //for (var j = 0; j < target.Height; j++)
            {
                for (var i = 0; i < target.Width; i++)
                {
                    var zSum = 0f;
                    for (var b = 0; b < dzdy.Height; b++)
                    {
                        for (var a = 0; a < dzdy.Width; a++)
                        {
                            var w = dzdy.GetValueFromCoordSafely(a, b);
                            var y = dedz.GetValueFromCoordPaddedWithZeroes(i - a, j - b);

                            var z = w * y;
                            zSum += z;
                        }
                    }

                    target.ChangeValueFromCoordSafely(
                        i,
                        j,
                        zSum,
                        invert_overwrite);

                    //if (overwrite)
                    //{
                    //    target.SetValueFromCoordSafely(
                    //        i,
                    //        j,
                    //        zSum
                    //        );
                    //}
                    //else
                    //{
                    //    target.AddValueFromCoordSafely(
                    //        i,
                    //        j,
                    //        zSum
                    //        );
                    //}
                }
            }
            ); //Parallel.For
        }

        #endregion

    }
}