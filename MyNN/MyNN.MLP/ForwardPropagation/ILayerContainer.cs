using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation
{
    public interface ILayerContainer
    {
        /// <summary>
        /// Конфигурация слоя
        /// </summary>
        ILayerConfiguration Configuration
        {
            get;
        }

        /// <summary>
        /// Очистить и записать значения NET и State
        /// </summary>
        void ClearAndPushNetAndState();

        /// <summary>
        /// Очистить значения NET и State
        /// </summary>
        void ClearNetAndState();

        /// <summary>
        /// Записать значения NET и State
        /// </summary>
        void PushNetAndState();

        /// <summary>
        /// Прочитать значения NET и State
        /// </summary>
        void PopNetAndState();

        /// <summary>
        /// Прочитать веса и биасы
        /// </summary>
        void PopWeightsAndBiases();

        /// <summary>
        /// Сохранить в контейнер значения для State
        /// </summary>
        void ReadInput(float[] data);

        /// <summary>
        /// Сохранить в контейнер веса из слоя сети и записать их
        /// </summary>
        void ReadWeightsAndBiasesFromLayer(ILayer layer);

        /// <summary>
        /// Сохранить веса в слой сети
        /// </summary>
        void WritebackWeightsAndBiasesToMLP(ILayer layer);

        /// <summary>
        /// Получить клонированное состояние State
        /// </summary>
        /// <returns></returns>
        ILayerState GetLayerState();


    }
}