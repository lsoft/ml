using System;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Other
{
    // ReSharper disable once InconsistentNaming
    public class rint
    {
        public int Min
        {
            get;
            private set;
        }

        public int Max
        {
            get;
            private set;
        }

        private readonly IRandomizer _randomizer;
        private readonly int _diff;

        public rint(IRandomizer randomizer, int min, int max)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (max < min)
            {
                throw new ArgumentException("max < min");
            }

            _randomizer = randomizer;

            Min = min;
            Max = max;

            _diff = max - min;
        }

        public int Sample()
        {
            var diff = _randomizer.Next(_diff);
            var result = this.Min + diff;

            return
                result;
        }

        public static implicit operator int(rint rf)
        {
            return
                rf.Sample();
        }


    }
}