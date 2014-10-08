using System.Collections.Generic;

namespace MyNN.Common.Data
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

        int InputLength        
        {
            get;
        }

        DataItem this[int i]
        {
            get;
        }

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

    }
}