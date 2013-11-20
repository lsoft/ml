using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public class SaltAndPepperNoiser : INoiser
    {
        private readonly double _zeroPercent;
        private readonly Random _random;

        public SaltAndPepperNoiser(
            ref int rndSeed,
            float noisePercent)
        {
            if (noisePercent < 0 || noisePercent >= 1f)
            {
                throw new ArgumentException("noisePercent");
            }

            _zeroPercent = (double)noisePercent;
            _random = new Random(++rndSeed);
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var r = new float[data.Length];

            for (var cc = 0; cc < data.Length; cc++)
            {
                r[cc] = _random.NextDouble() < _zeroPercent
                    ? (_random.NextDouble() < 0.5 ? 0f : 1f)
                    : data[cc];
            }

            return r;
        }
    }
}
