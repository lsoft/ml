using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public enum LayerTypeEnum
    {
        Input,
        FullConnected,
        Convolution,
        AvgPool,
        MaxPool
    }

    public interface ILayer
    {
        /// <summary>
        /// Тип слоя
        /// </summary>
        LayerTypeEnum Type
        {
            get;
        }

        IDimension SpatialDimension
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

        ILayerConfiguration GetConfiguration();

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
    }
}