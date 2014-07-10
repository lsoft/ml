using System.Collections.Generic;
using MyNN.MLP2.Structure.Layer.Factory.WeightLoader;

namespace MyNN.BoltzmannMachines.DBNInfo
{
    public interface IDBNInformation
    {
        /// <summary>
        /// Количество слоев в обученной DBN
        /// </summary>
        int LayerCount
        {
            get;
        }

        /// <summary>
        /// Размеры слоев обученной DBN
        /// </summary>
        int[] LayerSizes
        {
            get;
        }

        /// <summary>
        /// Получить загрузчик весов для автоенкодера
        /// </summary>
        /// <param name="layerIndex">Индекс слоя в сети (начинается с 1)</param>
        /// <param name="mlpLayersCount">Общее количество слоев в сети</param>
        /// <param name="encoderWeightLoader">Загрузчик весов для слоя кодирования</param>
        /// <param name="decoderWeightLoader">Загрузчик весов для слоя декодирования</param>
        void GetAutoencoderWeightLoaderForLayer(
            int layerIndex,
            int mlpLayersCount,
            out IWeightLoader encoderWeightLoader,
            out IWeightLoader decoderWeightLoader
            );

        /// <summary>
        /// Получить загрузчик весов для слоя сети
        /// </summary>
        /// <param name="layerIndex">Индекс слоя в сети (начинается с 1)</param>
        /// <param name="weightLoader">Загрузчик весов для слоя сети</param>
        void GetWeightLoaderForLayer(
            int layerIndex,
            out IWeightLoader weightLoader);
    }
}