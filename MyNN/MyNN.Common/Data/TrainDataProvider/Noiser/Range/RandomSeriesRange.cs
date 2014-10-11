using System;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.TrainDataProvider.Noiser.Range
{
    [Serializable]
    public class RandomSeriesRange : IRange
    {
        private readonly int _dataLength;
        private readonly IRandomizer _randomizer;

        public RandomSeriesRange(
            IRandomizer randomizer,
            int dataLength
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
            _dataLength = dataLength;
        }

        public bool[] GetIndexMask()
        {
            var minIncludeIndex = _randomizer.Next(_dataLength);
            var maxExcludeIndex = minIncludeIndex + _randomizer.Next(_dataLength - minIncludeIndex);
            
            var result = new bool[_dataLength];
            result.Fill((int index) => (minIncludeIndex <= index && index < maxExcludeIndex));

            return result;
        }
    }
}