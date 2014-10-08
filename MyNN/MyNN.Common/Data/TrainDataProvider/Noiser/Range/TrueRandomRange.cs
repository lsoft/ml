using System;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.TrainDataProvider.Noiser.Range
{
    public class TrueRandomRange : IRange
    {
        private readonly IRandomizer _randomizer;
        private readonly int _dataLength;
        private readonly float _threshold;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="randomizer"></param>
        /// <param name="dataLength"></param>
        /// <param name="threshold">Значение [0;1]. Чем выше значение - тем меньше шума применяется. 1 - не применять шум вообще, 0 - применять во всех ячеях (аналог FullRange)</param>
        public TrueRandomRange(
            IRandomizer randomizer,
            int dataLength,
            float threshold
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (threshold < 0 || threshold > 1f)
            {
                throw new ArgumentException("threshold < 0 || threshold > 1f");
            }

            _randomizer = randomizer;
            _dataLength = dataLength;
            _threshold = threshold;
        }

        public bool[] GetIndexMask()
        {
            var result = new bool[_dataLength];
            result.Fill(_randomizer.Next() > _threshold);

            return result;
        }

    }
}