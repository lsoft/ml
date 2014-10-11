using System;
using System.Collections.Generic;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.DataSetConverter
{
    /// <summary>
    /// Создает новый датасет, перемешивает его и отдает
    /// </summary>
    public class ShuffleDataSetConverter : IDataSetConverter
    {
        private readonly IRandomizer _randomizer;

        public ShuffleDataSetConverter(
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
        }

        public IDataSet Convert(IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            var cloned = new List<IDataItem>(beforeTransformation.Data);
            for (int i = 0; i < cloned.Count - 1; i++)
            {
                if (_randomizer.Next() >= 0.5d)
                {
                    var newIndex = _randomizer.Next(cloned.Count);

                    var tmp = cloned[i];
                    cloned[i] = cloned[newIndex];
                    cloned[newIndex] = tmp;
                }
            }

            var result = new DataSet(
                cloned);

            return result;
        }
    }
}