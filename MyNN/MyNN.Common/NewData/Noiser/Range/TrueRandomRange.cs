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
        /// <param name="randomizer">��������� ��������� �����</param>
        /// <param name="threshold">�������� [0;1]. ��� ���� �������� - ��� ������ ���� �����������. 1 - �� ��������� ��� ������, 0 - ��������� �� ���� ����� (������ FullRange)</param>
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