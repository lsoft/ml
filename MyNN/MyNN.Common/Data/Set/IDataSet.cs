using System.Collections.Generic;
using MyNN.Common.Data.Set.Item;

namespace MyNN.Common.Data.Set
{
    public interface IDataSet
    {
        List<IDataItem> Data
        {
            get;
        }

        bool IsAutoencoderDataSet
        {
            get;
        }

        int Count
        {
            get;
        }

        int InputLength
        {
            get;
        }

        /// <summary>
        /// Линейная нормализация [0;1]
        /// </summary>
        void Normalize(float bias = 0f);

        /// <summary>
        /// Гауссова нормализация
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        void GNormalize();

    }
}