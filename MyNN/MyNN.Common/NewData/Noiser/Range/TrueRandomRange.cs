using System;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.Noiser.Range
{
    [Serializable]
    public class TrueRandomRange : IRange
    {
        private readonly IRandomizer _randomizer;
        private readonly float _threshold;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="randomizer">Поставщик случайных чисел</param>
        /// <param name="threshold">Значение [0;1]. Чем выше значение - тем меньше шума применяется. 1 - не применять шум вообще, 0 - применять во всех ячеях (аналог FullRange)</param>
        public TrueRandomRange(
            IRandomizer randomizer,
            float threshold
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (threshold < 0 || threshold > 1f)
            {
                throw new ArgumentException("threshold < 0 || threshold > 1f");
            }

            _randomizer = randomizer;
            _threshold = threshold;
        }

        public bool[] GetIndexMask(int dataLength)
        {
            var result = new bool[dataLength];
            result.Fill(_randomizer.Next() > _threshold);

            return result;
        }

    }
}