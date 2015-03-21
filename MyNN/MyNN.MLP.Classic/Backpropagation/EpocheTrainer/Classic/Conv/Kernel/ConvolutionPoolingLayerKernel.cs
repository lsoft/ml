using System;
using System.Threading.Tasks;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel
{
    internal class ConvolutionPoolingLayerKernel
    {
        private readonly IConvolutionLayerConfiguration _currentLayerConfiguration;

        public ConvolutionPoolingLayerKernel(
            IConvolutionLayerConfiguration currentLayerConfiguration
            )
        {
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }

            _currentLayerConfiguration = currentLayerConfiguration;
        }

        public void CalculateOverwrite(
            IReferencedSquareFloat currentLayerNet, 
            IReferencedSquareFloat currentLayerDeDz,
            IReferencedSquareFloat previousLayerState,
            IReferencedSquareFloat nablaKernel,
            ref float nablaBias,

            IReferencedSquareFloat nextLayerDeDy,

            float learningRate
            )
        {
            if (currentLayerNet == null)
            {
                throw new ArgumentNullException("currentLayerNet");
            }
            if (currentLayerDeDz == null)
            {
                throw new ArgumentNullException("currentLayerDeDz");
            }
            if (previousLayerState == null)
            {
                throw new ArgumentNullException("previousLayerState");
            }
            if (nablaKernel == null)
            {
                throw new ArgumentNullException("nablaKernel");
            }
            if (nextLayerDeDy == null)
            {
                throw new ArgumentNullException("nextLayerDeDy");
            }

            //////вычисляем значение ошибки (dE/dz) апсемплингом с пулинг слоя
            //////(реализация неэффективная! переделать!)
            //////умножением на _nextLayerConfiguration.ScaleFactor может
            //////несколько раз браться одно и то же значение из nextLayerDeDz
            //////можно попробовать предрассчитать индексы и брать готовые
            ////for (var j = 0; j < currentLayerState.Height; j++)
            ////{
            ////    for (var i = 0; i < currentLayerState.Width; i++)
            ////    {
            ////        var jp = (int)(j * _nextLayerConfiguration.ScaleFactor);
            ////        var ip = (int)(i * _nextLayerConfiguration.ScaleFactor);

            ////        var v = nextLayerDeDz.GetValueFromCoordSafely(ip, jp);

            ////        //v *= _nextLayerConfiguration.InverseScaleFactor*_nextLayerConfiguration.InverseScaleFactor;

            ////        deDz.SetValueFromCoordSafely(i, j, v);
            ////    }
            ////}

            var locker = new object();
            var nb = 0f;

            //считаем наблу
            Parallel.For(0, nablaKernel.Height, b =>
            //for (var b = 0; b < nablaKernel.Height; b++)
            {
                for (var a = 0; a < nablaKernel.Width; a++)
                {
                    var dEdw_ab = 0f; //kernel
                    var dEdb_ab = 0f; //bias

                    for (var j = 0; j < currentLayerNet.Height; j++)
                    {
                        for (var i = 0; i < currentLayerNet.Width; i++)
                        {
                            var dedy = nextLayerDeDy.GetValueFromCoordSafely(i, j);
                            float z = currentLayerNet.GetValueFromCoordSafely(i, j);
                            float dydz = _currentLayerConfiguration.LayerActivationFunction.ComputeFirstDerivative(z);
                            var dEdz_ij = dedy * dydz;

                            currentLayerDeDz.SetValueFromCoordSafely(i, j, dEdz_ij);

                            //var dEdz_ij = deDz.GetValueFromCoordSafely(i, j);

                            var y = previousLayerState.GetValueFromCoordSafely(
                                i + a,
                                j + b
                                );

                            var w_mul = dEdz_ij * y * learningRate;
                            dEdw_ab += w_mul;

                            var b_mul = dEdz_ij * learningRate;
                            dEdb_ab += b_mul;

                        }
                    }

                    //вычислено dEdw_ab, dEdb_ab
                    nablaKernel.SetValueFromCoordSafely(a, b, dEdw_ab);
                    //nablaBias = dEdb_ab * learningRate;

                    lock (locker)
                    {
                        nb += dEdb_ab * learningRate;
                    }
                }
            }
            ); //Parallel.For

            nablaBias += nb;
        }
    }
}
