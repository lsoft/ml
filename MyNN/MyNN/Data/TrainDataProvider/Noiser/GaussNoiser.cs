using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.Distributions;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public class GaussNoiser : INoiser
    {
        private readonly bool _isNeedToClamp01;
        private readonly Normal _random;

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
                var v = data[cc] + (float)_random.Sample();

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

                r[cc] = v;
            }

            return r;
        }
    }
}
