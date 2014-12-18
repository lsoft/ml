using System;
using MyNN.Common.NewData.Noiser.Range;

namespace MyNN.Common.NewData.Noiser
{
    [Serializable]
    public class GaussNoiser : INoiser
    {
        private readonly float _stddev;
        private readonly bool _isNeedToClamp01;
        private readonly Random _random;
        private readonly IRange _range;

        /// <summary>
        /// Гауссов зашумитель
        /// </summary>
        /// <param name="stddev">Стандартное отклонение гауссово шума</param>
        /// <param name="isNeedToClamp01">Нужно ли ограничивать значение после зашумления интервалом [0;1]</param>
        /// <param name="range">На каком диапазоне зашумлять</param>
        public GaussNoiser(
            float stddev,
            bool isNeedToClamp01,
            IRange range)
        {
            if (range == null)
            {
                throw new ArgumentNullException("range");
            }

            _stddev = stddev;
            _isNeedToClamp01 = isNeedToClamp01;
            _range = range;

            _random = new Random();
        }

        /// <summary>
        /// Гауссов зашумитель
        /// </summary>
        /// <param name="stddev">Стандартное отклонение гауссово шума</param>
        /// <param name="isNeedToClamp01">Нужно ли ограничивать значение после зашумления интервалом [0;1]</param>
        public GaussNoiser(
            float stddev,
            bool isNeedToClamp01)
        {
            _stddev = stddev;
            _isNeedToClamp01 = isNeedToClamp01;
            _range = null;

            _random = new Random();
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var gaussRandom = new Normal(0, _stddev);
            gaussRandom.RandomSource = _random;

            var r = new float[data.Length];

            var range = _range ?? new FullRange();

            var mask = range.GetIndexMask(data.Length);
            for (var cc = 0; cc < data.Length; cc++)
            {
                var v = data[cc];

                if (mask[cc])
                {
                    v += (float)gaussRandom.Sample();

                    if (_isNeedToClamp01)
                    {
                        if (v < 0f)
                        {
                            v = 0f;
                        }

                        if (v > 1f)
                        {
                            v = 1f;
                        }
                    }
                }

                r[cc] = v;
            }

            return r;
        }
    }
}
