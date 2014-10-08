using System;
using MathNet.Numerics.Distributions;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;

namespace MyNN.Common.Data.TrainDataProvider.Noiser
{
    public class GaussNoiser : INoiser
    {
        private readonly bool _isNeedToClamp01;
        private readonly Normal _random;
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

            _isNeedToClamp01 = isNeedToClamp01;
            _random = new Normal(0, stddev);
            _range = range;
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
            _isNeedToClamp01 = isNeedToClamp01;
            _random = new Normal(0, stddev);
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
                    v += (float)_random.Sample();

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
