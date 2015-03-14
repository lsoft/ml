using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel
{
    public class AvgPoolingFullConnectedKernel
    {
        private readonly ILayerConfiguration _layerConfiguration;

        public AvgPoolingFullConnectedKernel(
            ILayerConfiguration layerConfiguration
            )
        {
            if (layerConfiguration == null)
            {
                throw new ArgumentNullException("layerConfiguration");
            }

            _layerConfiguration = layerConfiguration;
        }

        public void Calculate(
            int fmiNeuronIndex,
            IReferencedSquareFloat currentLayerDeDz,
            float[] nextLayerDeDy
            )
        {
            if (currentLayerDeDz == null)
            {
                throw new ArgumentNullException("currentLayerDeDz");
            }
            if (nextLayerDeDy == null)
            {
                throw new ArgumentNullException("nextLayerDeDy");
            }

            //вычисляем значение ошибки (dE/dz) суммированием по след слою
            //а след слой - полносвязный и у него веса к каждому пулинг
            //нейрону может быть разным
            var neuronIndex = fmiNeuronIndex;
            for (var i = 0; i < _layerConfiguration.SpatialDimension.Width; i++)
            {
                for (var j = 0; j < _layerConfiguration.SpatialDimension.Height; j++)
                {
                    //var accDeDy = new KahanAlgorithm.Accumulator();
                    //for (var q = 0; q < nextLayerDeDz.Length; q++)
                    //{
                    //    var nextWeightIndex = ComputeWeightIndex(_layerConfiguration.TotalNeuronCount, q) + neuronIndex; //не векторизуется:(
                    //    var wijk = nextLayerWeights[nextWeightIndex];

                    //    var ndedz = nextLayerDeDz[q];

                    //    var multiplied = wijk * ndedz;

                    //    KahanAlgorithm.AddElement(ref accDeDy, multiplied);
                    //}

                    var dedy = nextLayerDeDy[neuronIndex];

                    //var dedz = accDeDy.Sum * 1;
                    var dedz = dedy * 1;
                    //для avg pooling dedy тоже самое что и  dedz, так как нет функции активации
                    //(или можно сказать функция активации линейна и ее производная равна 1, что
                    //и показано для наглядности в формуле)

                    currentLayerDeDz.SetValueFromCoordSafely(i, j, dedz);

                    neuronIndex++;
                }
            }

            //здесь должен быть расчет наблы по весам, но весов у пулинга нету
            //фактически это означает, что пулинг слой ничему не "учится"

            //произведение кронекера на единичную матрицу (или проще - апсемплинг dedz)
            //делаем на слое свертки

        }

    }
}
