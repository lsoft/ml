using System;
using System.Linq;
using MyNN.Common.NewData.Item;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    [Serializable]
    public class BinarizeDataItemTransformation : IDataItemTransformation
    {
        private readonly IRandomizer _randomizer;
        private readonly IDataItemFactory _dataItemFactory;

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    false;
            }
        }

        public BinarizeDataItemTransformation(
            IRandomizer randomizer,
            IDataItemFactory dataItemFactory
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            _randomizer = randomizer;
            _dataItemFactory = dataItemFactory;
        }

        public IDataItem Transform(IDataItem before)
        {
            if (before == null)
            {
                throw new ArgumentNullException("before");
            }

            if (before.Input.Any(j => j < 0f || j > 1f))
            {
                throw new InvalidOperationException("Данные не нормализованы в диапазон [0;1]");
            }

            var bi = before.Input.ToList().ConvertAll(j => (_randomizer.Next() < j) ? 1f : 0f);

            var newItem = _dataItemFactory.CreateDataItem(
                bi.ToArray(),
                before.Output
                );

            return
                newItem;
        }

    }
}