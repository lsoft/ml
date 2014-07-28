using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Randomizer;

namespace MyNN.Data.DataSetConverter
{
    /// <summary>
    /// Ѕинаризует данные в датасете
    /// (1 с веро€тностью значени€)
    /// ≈сли данные не нормализованы в диапазон [0;1], генерируетс€ исключение
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

            var cloned = new List<DataItem>();
            foreach (var di in beforeTransformation.Data)
            {
                if (di.Input.Any(j => j < 0f || j > 1f))
                {
                    throw new InvalidOperationException("ƒанные не нормализованы в диапазон [0;1]");
                }

                var bi = di.Input.ToList().ConvertAll(j => (_randomizer.Next() < j) ? 1f : 0f);

                var ndi = new DataItem(bi.ToArray(), di.Output);
                cloned.Add(ndi);
            }

            var result = new DataSet(
                cloned);

            return result;
        }
    }
}