﻿using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayer
    {
        /// <summary>
        /// Тип слоя
        /// </summary>
        LayerTypeEnum Type
        {
            get;
        }

        /// <summary>
        /// Всего нейронов в слое
        /// </summary>
        int TotalNeuronCount
        {
            get;
        }

        INeuron[] Neurons
        {
            get;
        }

        IFunction LayerActivationFunction
        {
            get;
        }

        string GetLayerInformation();

        /// <summary>
        /// Получить массив клонированных весов всех нейронов сети
        /// </summary>
        void GetClonedWeights(
            out float[] weights,
            out float[] biases
            );

        /// <summary>
        /// Записать веса в слой
        /// </summary>
        void SetWeights(
            float[] weights,
            float[] biases
            );

        IDimension SpatialDimension
        {
            get;
        }

        ILayerConfiguration GetConfiguration();

    }
}