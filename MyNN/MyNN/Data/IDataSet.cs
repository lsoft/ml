using System.Collections.Generic;
using MyNN.Randomizer;

namespace MyNN.Data
{
    public interface IDataSet : IEnumerable<DataItem>
    {
        List<DataItem> Data
        {
            get;
        }

        bool IsAuencoderDataSet
        {
            get;
        }

        int Count
        {
            get;
        }

        DataItem this[int i]
        {
            get;
        }

        IDataSet ConvertToAutoencoder();

        List<float[]> GetInputPart();

        /// <summary>
        /// Линейная нормализация [0;1]
        /// </summary>
        void Normalize(float bias = 0f);

        /// <summary>
        /// Гауссова нормализация
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        void GNormalize();

        /// <summary>
        /// Создает новый датасет, перемешивает его и отдает
        /// </summary>
        /// <returns></returns>
        IDataSet CreateShuffledDataSet(
            IRandomizer randomizer);
        
        /// <summary>
        /// Бинаризует данные в датасете
        /// (1 с вероятностью значения)
        /// Если данные не нормализованы в диапазон [0;1], генерируется исключение
        /// </summary>
        /// <returns></returns>
        IDataSet Binarize(
            IRandomizer randomizer);
    }
}