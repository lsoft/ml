﻿using System;
using MyNN.Common.NewData.Noiser.Range;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.Noiser
{
    [Serializable]
    public class MultiplierNoiser : INoiser
    {
        private readonly float _applyPercent;
        private readonly IRandomizer _randomizer;
        private readonly IRange _range;

        public MultiplierNoiser(
            IRandomizer randomizer,
            float applyPercent,
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
            if (applyPercent <= 0 || applyPercent > 1f)
            {
                throw new ArgumentException("applyPercent");
            }

            _applyPercent = applyPercent;
            _randomizer = randomizer;
            _range = range;
        }

        public MultiplierNoiser(
            IRandomizer randomizer,
            float applyPercent)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (applyPercent <= 0 || applyPercent > 1f)
            {
                throw new ArgumentException("applyPercent");
            }

            _applyPercent = applyPercent;
            _randomizer = randomizer;
            _range = null;
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var r = new float[data.Length];

            var range = _range ?? new FullRange();

            var mask = range.GetIndexMask(data.Length);

            for (var cc = 0; cc < data.Length; cc++)
            {
                var v = data[cc];

                if (mask[cc])
                {
                    if (_randomizer.Next() < _applyPercent)
                    {
                        v *= _randomizer.Next();
                    }
                }

                r[cc] = v;
            }

            return r;
        }
    }
}
