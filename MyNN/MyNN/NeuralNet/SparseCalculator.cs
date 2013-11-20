using System;
using System.Linq;
using MyNN.Data;
using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet
{
    /// <summary>
    /// Вычисляет количество нулей в среднем по датасету на конечном слое нейросети
    /// </summary>
    public class SparseCalculator
    {
        /// <summary>
        /// Просчет доли нулей
        /// </summary>
        /// <param name="network">Сеть, через которую прогоняются данные</param>
        /// <param name="dataset">Проверочные данные (обычно валидационные)</param>
        /// <returns>Доля нулей по всем нейронам выходного слоя и по всему датасету (0-1)</returns>
        public float Calculate(
            MultiLayerNeuralNetwork network,
            DataSet dataset)
        {
            if (network == null)
            {
                throw new ArgumentNullException("network");
            }
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }

            var sparseddata = network.ComputeOutput(dataset.GetInputPart());
            var totalZero = sparseddata.Sum(j => j.Count(k => k < float.Epsilon));
            var sparsePart = totalZero / (float)sparseddata.Count / (float)sparseddata[0].Length;

            return sparsePart;
        }
    }
}
