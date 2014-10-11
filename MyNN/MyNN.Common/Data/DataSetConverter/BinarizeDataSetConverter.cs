using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.DataSetConverter
{
    /// <summary>
    /// ���������� ������ � ��������
    /// (1 � ������������ ��������)
    /// ���� ������ �� ������������� � �������� [0;1], ������������ ����������
    /// </summary>
    public class BinarizeDataSetConverter : IDataSetConverter
    {
        private readonly IRandomizer _randomizer;

        public BinarizeDataSetConverter(
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
        }

        public IDataSet Convert(
            IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            var cloned = new List<IDataItem>();
            foreach (var di in beforeTransformation.Data)
            {
                if (di.Input.Any(j => j < 0f || j > 1f))
                {
                    throw new InvalidOperationException("������ �� ������������� � �������� [0;1]");
                }

                var bi = di.Input.ToList().ConvertAll(j => (_randomizer.Next() < j) ? 1f : 0f);

                var ndi = new DenseDataItem(bi.ToArray(), di.Output);
                cloned.Add(ndi);
            }

            var result = new DataSet(
                cloned);

            return result;
        }
    }
}