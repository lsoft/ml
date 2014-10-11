using System.Collections.Generic;
using MyNN.Common.Data.Set.Item;

namespace MyNN.Common.Data.Set
{
    public interface IDataSet : IEnumerable<IDataItem>
    {
        List<IDataItem> Data
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

        IDataItem this[int i]
        {
            get;
        }

        List<float[]> GetInputPart();

        /// <summary>
        /// �������� ������������ [0;1]
        /// </summary>
        void Normalize(float bias = 0f);

        /// <summary>
        /// �������� ������������
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        void GNormalize();

    }
}