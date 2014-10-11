using System;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.TrainDataProvider.Noiser
{
    [Serializable]
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
            _range = null;
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var r = new float[data.Length];

            var range = _range ?? new FullRange(data.Length);

            var mask = range.GetIndexMask();

            for (var cc = 0; cc < data.Length; cc++)
            {
                var v = data[cc];

                if (mask[cc])
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
