using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.MLP2.Randomizer;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public class ZeroMaskingNoiser : INoiser
    {
        private readonly float _zeroPercent;
        private readonly IRandomizer _randomizer;
        private readonly IRange _range;

        public ZeroMaskingNoiser(
            IRandomizer randomizer,
            float zeroPercent,
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
            if (zeroPercent < 0 || zeroPercent >= 1f)
            {
                throw new ArgumentException("zeroPercent");
            }

            _zeroPercent = zeroPercent;
            _randomizer = randomizer;
            _range = range;
        }

        public ZeroMaskingNoiser(
            IRandomizer randomizer,
            float zeroPercent)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (zeroPercent < 0 || zeroPercent >= 1f)
            {
                throw new ArgumentException("zeroPercent");
            }

            _zeroPercent = zeroPercent;
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

            int min = 0, max = data.Length;
            _range.GetIndexes(data.Length, out min, out max);

            for (var cc = 0; cc < data.Length; cc++)
            {
                var v = data[cc];

                if (cc >= min && cc < max)
                {
                    if (_randomizer.Next() < _zeroPercent)
                    {
                        v = 0f;
                    }
                }

                r[cc] = v;
            }

            return r;
        }
    }
}
