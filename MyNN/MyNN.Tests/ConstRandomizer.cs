using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Common.Randomizer;

namespace MyNN.Tests
{
    [Serializable]
    internal class ConstRandomizer : IRandomizer
    {
        private readonly float _fromZeroToOne;

        public ConstRandomizer(float fromZeroToOne)
        {
            _fromZeroToOne = fromZeroToOne;
        }

        public int Next(int maxValue)
        {
            return (int)(_fromZeroToOne*maxValue);
        }

        public float Next()
        {
            return _fromZeroToOne;
        }

        public void NextBytes(byte[] buffer)
        {
            if (buffer == null)
            {
                throw new ArgumentNullException("buffer");
            }

            var v = (byte) (_fromZeroToOne*256);

            for (var cc = 0; cc < buffer.Length; cc++)
            {
                buffer[cc] = v;
            }
        }
    }
}
