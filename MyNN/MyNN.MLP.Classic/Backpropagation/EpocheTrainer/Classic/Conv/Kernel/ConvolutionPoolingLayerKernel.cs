using System;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel
{
    internal class ConvolutionPoolingLayerKernel
    {
        private readonly ILayer _layer;
        private readonly IAvgPoolingLayer _nextLayer;

        public ConvolutionPoolingLayerKernel(
            ILayer layer,
            IAvgPoolingLayer nextLayer
            )
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }
            if (nextLayer == null)
            {
                throw new ArgumentNullException("nextLayer");
            }

            _layer = layer;
            _nextLayer = nextLayer;
        }

        public void CalculateOverwrite(
            IReferencedSquareFloat currentLayerState,
            IReferencedSquareFloat previousLayerState,
            IReferencedSquareFloat deDz,
            IReferencedSquareFloat nextLayerDeDz,

            IReferencedSquareFloat nablaKernel,
            ref float nablaBias,

            float learningRate
            )
        {
            if (currentLayerState == null)
            {
                throw new ArgumentNullException("currentLayerState");
            }
            if (previousLayerState == null)
            {
                throw new ArgumentNullException("previousLayerState");
            }
            if (nablaKernel == null)
            {
                throw new ArgumentNullException("nablaKernel");
            }
            if (deDz == null)
            {
                throw new ArgumentNullException("deDz");
            }
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }

            //вычисляем значение ошибки (dE/dz) апсемплингом с пулинг слоя
            //(реализация неэффективная! переделать!)
            for (var j = 0; j < currentLayerState.Height; j++)
            {
                for (var i = 0; i < currentLayerState.Width; i++)
                {
                    var jp = (int)(j * _nextLayer.ScaleFactor);
                    var ip = (int)(i * _nextLayer.ScaleFactor);

                    var v = nextLayerDeDz.GetValueFromCoordSafely(ip, jp);

                    //v *= _nextLayer.InverseScaleFactor*_nextLayer.InverseScaleFactor;

                    deDz.SetValueFromCoordSafely(i, j, v);
                }
            }

            //считаем наблу
            for (var a = 0; a < nablaKernel.Width; a++)
            {
                for (var b = 0; b < nablaKernel.Height; b++)
                {
                    var dEdw_ab = 0f; //kernel
                    var dEdb_ab = 0f; //bias

                    for (var i = 0; i < currentLayerState.Width; i++)
                    {
                        for (var j = 0; j < currentLayerState.Height; j++)
                        {
                            var dEdz_ij = deDz.GetValueFromCoordSafely(i, j);

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

                    //вычислено dEdw_ab, dEdz_ab
                    nablaKernel.SetValueFromCoordSafely(a, b, dEdw_ab);
                    nablaBias = dEdb_ab * learningRate;
                }
            }

        }

    }
}
