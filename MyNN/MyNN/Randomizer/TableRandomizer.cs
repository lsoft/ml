using System;
using System.Threading;

namespace MyNN.Randomizer
{
    [Serializable]
    public class TableRandomizer : IRandomizer
    {
        private readonly object _lockObject = new object();

        private readonly double[] _table = new[]
        {
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0
        };

        private int _currentValue = 0;

        private int _autoIncrementCurrentValue
        {
            get
            {
                var result = Interlocked.Increment(ref _currentValue);

                return
                    result;
            }
        }

        public int Next(int maxValue)
        {
            lock (_lockObject)
            {
                var index = _autoIncrementCurrentValue % _table.Length;

                var result = (int) (_table[index] * maxValue);

                return result;
            }
        }

        public float Next()
        {
            lock (_lockObject)
            {

                var index = _autoIncrementCurrentValue%_table.Length;

                var result = (float)_table[index];

                return result;
            }
        }

        public void NextBytes(byte[] buffer)
        {
            if (buffer == null)
            {
                throw new ArgumentNullException("buffer");
            }

            lock (_lockObject)
            {
                for (var i = 0; i < buffer.Length; i++)
                {
                    buffer[i] = (byte)(_autoIncrementCurrentValue % 256);
                }
            }
        }
    }
}