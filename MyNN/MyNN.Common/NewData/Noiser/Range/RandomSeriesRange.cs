using System;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.Noiser.Range
{
    [Serializable]
    public class RandomSeriesRange : IRange
    {
        private readonly IRandomizer _randomizer;

        public RandomSeriesRange(
            IRandomizer randomizer
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
        }

        public bool[] GetIndexMask(int dataLength)
        {
            var minIncludeIndex = _randomizer.Next(dataLength);
            var maxExcludeIndex = minIncludeIndex + _randomizer.Next(dataLength - minIncludeIndex);
            
            var result = new bool[dataLength];
            result.Fill((int index) => (minIncludeIndex <= index && index < maxExcludeIndex));

            return result;
        }
    }
}