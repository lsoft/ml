using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public class ZeroMaskingNoiser : INoiser
    {
        private readonly double _zeroPercent;
        private readonly Random _random;

        public ZeroMaskingNoiser(
            ref int rndSeed,
            float zeroPercent)
        {
            if (zeroPercent < 0 || zeroPercent >= 1f)
            {
                throw new ArgumentException("zeroPercent");
            }

            _zeroPercent = (double)zeroPercent;
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
                r[cc] = _random.NextDouble() < _zeroPercent ? 0f : data[cc];
            }

            return r;
        }
    }
}
