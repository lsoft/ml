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
        /// �������� ������������ [0;1]
        /// </summary>
        void Normalize(float bias = 0f);

        /// <summary>
        /// �������� ������������
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        void GNormalize();

        /// <summary>
        /// ������� ����� �������, ������������ ��� � ������
        /// </summary>
        /// <returns></returns>
        IDataSet CreateShuffledDataSet(
            IRandomizer randomizer);
        
        /// <summary>
        /// ���������� ������ � ��������
        /// (1 � ������������ ��������)
        /// ���� ������ �� ������������� � �������� [0;1], ������������ ����������
        /// </summary>
        /// <returns></returns>
        IDataSet Binarize(
            IRandomizer randomizer);
    }
}