using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.MLP2.Randomizer;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public class DistanceChangeNoiser : INoiser
    {
        private readonly float _changePercent;
        private readonly int _maxDistance;
        private readonly IRandomizer _randomizer;
        private readonly IRange _range;

        public DistanceChangeNoiser(
            IRandomizer randomizer,
            float changePercent,
            int maxDistance,
            IRange range)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (range == null)
            {
                throw new ArgumentNullException("range");
            }
            if (changePercent <= 0 || changePercent > 1f)
            {
                throw new ArgumentException("changePercent");
            }
            if (maxDistance <= 0)
            {
                throw new ArgumentException("maxDistance");
            }

            _changePercent = changePercent;
            _maxDistance = maxDistance;
            _randomizer = randomizer;
            _range = range;
        }

        public DistanceChangeNoiser(
            IRandomizer randomizer,
            float changePercent,
            int maxDistance)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (changePercent <= 0 || changePercent > 1f)
            {
                throw new ArgumentException("changePercent");
            }
            if (maxDistance <= 0)
            {
                throw new ArgumentException("maxDistance");
            }

            _changePercent = changePercent;
            _maxDistance = maxDistance;
            _randomizer = randomizer;
            _range = new FullRange();
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var r = new float[data.Length];
            data.CopyTo(r, 0);

            int min = 0, max = data.Length;
            _range.GetIndexes(data.Length, out min, out max);

            for (var cc = 0; cc < data.Length; cc++)
            {
                if (cc >= min && cc < max)
                {
                    if (_randomizer.Next() < _changePercent)
                    {
                        var newIndex= cc + (_randomizer.Next(_maxDistance * 2) - _maxDistance);

                        if (newIndex >= 0 && newIndex < data.Length)
                        {
                            var v0 = data[cc];
                            var v1 = data[newIndex];

                            r[cc] = v1;
                            r[newIndex] = v0;
                        }
                    }
                }

            }

            return r;
        }
    }
}
