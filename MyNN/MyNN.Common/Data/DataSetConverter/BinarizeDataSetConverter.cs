using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Dense;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.DataSetConverter
{
    /// <summary>
    /// Ѕинаризует данные в датасете
    /// (1 с веро€тностью значени€)
    /// ≈сли данные не нормализованы в диапазон [0;1], генерируетс€ исключение
    /// </summary>
    public class BinarizeDataSetConverter : IDataSetConverter
    {
        private readonly IRandomizer _randomizer;
        private readonly IDataItemFactory _dataItemFactory;

        public BinarizeDataSetConverter(
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
                    throw new InvalidOperationException("ƒанные не нормализованы в диапазон [0;1]");
                }

                var bi = di.Input.ToList().ConvertAll(j => (_randomizer.Next() < j) ? 1f : 0f);

                var ndi = _dataItemFactory.CreateDataItem(
                    bi.ToArray(),
                    di.Output
                    );
                
                cloned.Add(ndi);
            }

            var result = new DataSet(
                cloned);

            return result;
        }
    }
}