using System;
using MyNN.Randomizer;

namespace MyNN.Data.TrainDataProvider.Noiser.Range
{
    public class RandomRange : IRange
    {
        private readonly IRandomizer _randomizer;

        public RandomRange(
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
        }

        public void GetIndexes(
            int length,
            out int minIncludeIndex,
            out int maxExcludeIndex)
        {
            minIncludeIndex = _randomizer.Next(length);
            maxExcludeIndex = minIncludeIndex + _randomizer.Next(length - minIncludeIndex);
        }
    }
}