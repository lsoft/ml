using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.MLP2.Randomizer;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public class SaltAndPepperNoiser : INoiser
    {
        private readonly double _applyPercent;
        private readonly IRandomizer _randomizer;
        private readonly IRange _range;

        public SaltAndPepperNoiser(
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
            if (applyPercent < 0 || applyPercent >= 1f)
            {
                throw new ArgumentException("applyPercent");
            }

            _applyPercent = applyPercent;
            _randomizer = randomizer;
            _range = range;
        }

        public SaltAndPepperNoiser(
            IRandomizer randomizer,
            float applyPercent)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (applyPercent < 0 || applyPercent >= 1f)
            {
                throw new ArgumentException("applyPercent");
            }

            _applyPercent = applyPercent;
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
                    if (_randomizer.Next() < _applyPercent)
                    {
                        v = _randomizer.Next() < 0.5f ? 0f : 1f;
                    }
                }

                r[cc] = v;
            }

            return r;
        }
    }
}
