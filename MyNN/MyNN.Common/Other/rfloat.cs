using System;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Other
{
// ReSharper disable once InconsistentNaming
    public class rfloat
    {
        public float Min
        {
            get;
            private set;
        }

        public float Max
        {
            get;
            private set;
        }

        private readonly IRandomizer _randomizer;
        private readonly float _diff;

        public rfloat(IRandomizer randomizer, float min, float max)
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

        public float Sample()
        {
            var rnd = _randomizer.Next();
            var result = this.Min + rnd * _diff;
            
            return
                result;
        }

        public static implicit operator float(rfloat rf)
        {
            return
                rf.Sample();
        }

    
    }
}
