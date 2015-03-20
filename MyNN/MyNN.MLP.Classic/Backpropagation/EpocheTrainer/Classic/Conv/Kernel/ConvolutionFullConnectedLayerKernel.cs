using System;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel
{
    internal class ConvolutionFullConnectedLayerKernel
    {
        private readonly ILayerConfiguration _layerConfiguration;

        public ConvolutionFullConnectedLayerKernel(
            ILayerConfiguration layerConfiguration
            )
        {
            if (layerConfiguration == null)
            {
                throw new ArgumentNullException("layerConfiguration");
            }

            _layerConfiguration = layerConfiguration;
        }

        public void CalculateOverwrite(
            int fmiNeuronIndex,
            IReferencedSquareFloat currentLayerNet,
            IReferencedSquareFloat previousLayerState,
            IReferencedSquareFloat deDz,
            IReferencedSquareFloat nablaKernel,
            ref float nablaBias,

            float[] nextLayerDeDy,

            float learningRate
            )
        {
            if (currentLayerNet == null)
            {
                throw new ArgumentNullException("currentLayerNet");
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
            if (deDz == null)
            {
                throw new ArgumentNullException("deDz");
            }

            var neuronIndex = fmiNeuronIndex;

            //возможно, стоит объединить оба цикла (вычисление dedz и обоих набл)
            //в один цикл, таким образом, минимизируя обращения к памяти
            //в сишарпе возможно, разницы никакой нет, но в opencl
            //это будет дополнительное обращение к глобальной памяти, что нехорошо

            //вычисляем значение ошибки (dE/dz) суммированием по след слою
            //а след слой - полносвязный и у него веса к каждому сверточному
            //нейрону может быть разным
            for (var i = 0; i < currentLayerNet.Width; i++)
            {
                for (var j = 0; j < currentLayerNet.Height; j++)
                {
                    ////счет dedy для следующего полносвязного слоя
                    ////neuronIndex каждый раз изменяется
                    //var accDeDy = new KahanAlgorithm.Accumulator();
                    //for (var q = 0; q < nextLayerDeDz.Length; q++)
                    //{
                    //    var nextWeightIndex = ComputeWeightIndex(_layerConfiguration.TotalNeuronCount, q) + neuronIndex; //не векторизуется:(
                    //    var wijk = nextLayerWeights[nextWeightIndex];

                    //    var ndedz = nextLayerDeDz[q];

                    //    var dedy = wijk * ndedz;

                    //    KahanAlgorithm.AddElement(ref accDeDy, dedy);
                    //}

                    var dedy = nextLayerDeDy[neuronIndex];

                    var z = currentLayerNet.GetValueFromCoordSafely(i, j);
                    var dydz = _layerConfiguration.LayerActivationFunction.ComputeFirstDerivative(z);

                    //var dedz = accDeDy.Sum * dydz;
                    var dedz = dedy * dydz;

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

                            var b_mul = dEdz_ij * 1 * learningRate;
                            dEdb_ab += b_mul;

                        }
                    }

                    //вычислено dEdw_ab, dEdb_ab
                    nablaKernel.SetValueFromCoordSafely(a, b, dEdw_ab);
                    nablaBias = dEdb_ab * learningRate;
                }
            }

        }

    }
}
