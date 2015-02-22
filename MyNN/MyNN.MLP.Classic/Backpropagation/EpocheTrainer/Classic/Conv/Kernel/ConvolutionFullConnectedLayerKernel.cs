using System;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel
{
    internal class ConvolutionFullConnectedLayerKernel
    {
        private readonly ILayer _layer;

        public ConvolutionFullConnectedLayerKernel(
            ILayer layer
            )
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            _layer = layer;
        }

        public void CalculateOverwrite(
            int fmiNeuronIndex,
            IReferencedSquareFloat currentLayerNet,
            IReferencedSquareFloat currentLayerState,
            IReferencedSquareFloat previousLayerState,
            IReferencedSquareFloat deDz,
            float[] nextLayerDeDz,
            IReferencedSquareFloat nablaKernel,
            ref float nablaBias,

            float learningRate,
            float[] nextLayerWeights
            )
        {
            if (currentLayerNet == null)
            {
                throw new ArgumentNullException("currentLayerNet");
            }
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
            if (nextLayerWeights == null)
            {
                throw new ArgumentNullException("nextLayerWeights");
            }
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }

            //вычисляем значение ошибки (dE/dz) суммированием по след слою
            //а след слой - полносвязный и у него веса к каждому сверточному
            //нейрону может быть разным
            var neuronIndex = fmiNeuronIndex;
            for (var i = 0; i < currentLayerNet.Width; i++)
            {
                for (var j = 0; j < currentLayerNet.Height; j++)
                {
                    var accDeDy = new KahanAlgorithm.Accumulator();
                    for (var q = 0; q < nextLayerDeDz.Length; q++)
                    {
                        var nextWeightIndex = ComputeWeightIndex(_layer.TotalNeuronCount, q) + neuronIndex; //не векторизуется:(
                        var wijk = nextLayerWeights[nextWeightIndex];

                        var ndedz = nextLayerDeDz[q];

                        var multiplied = wijk * ndedz;

                        KahanAlgorithm.AddElement(ref accDeDy, multiplied);
                    }

                    var z = currentLayerNet.GetValueFromCoordSafely(i, j);
                    var dydz = _layer.LayerActivationFunction.ComputeFirstDerivative(z);

                    var dedz = accDeDy.Sum * dydz;

                    deDz.SetValueFromCoordSafely(i, j, dedz);

                    neuronIndex++;
                }
            }

            //считаем наблу
            for (var a = 0; a < nablaKernel.Width; a++)
            {
                for (var b = 0; b < nablaKernel.Height; b++)
                {
                    var dEdw_ab = 0f; //kernel
                    var dEdb_ab = 0f; //bias

                    for (var i = 0; i < currentLayerNet.Width; i++)
                    {
                        for (var j = 0; j < currentLayerNet.Height; j++)
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

        private static int ComputeWeightIndex(
            int previousLayerNeuronCount,
            int neuronIndex)
        {
            return
                previousLayerNeuronCount * neuronIndex;
        }

    }
}
