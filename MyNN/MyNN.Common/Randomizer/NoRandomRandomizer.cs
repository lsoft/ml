using System;
using System.Threading;

namespace MyNN.Common.Randomizer
{
    [Serializable]
    public class NoRandomRandomizer : IRandomizer
    {
        private readonly object _lockObject = new object();
        private int _currentValue;

        private int _autoIncrementCurrentValue
        {
            get
            {
                var result = Interlocked.Increment(ref _currentValue);
                
                return
                    result;
            }
        }

        public NoRandomRandomizer()
        {
            _currentValue = 0;
        }

        public int Next(int maxValue)
        {
            lock (_lockObject)
            {
                return
                    _autoIncrementCurrentValue%maxValue;
            }
        }

        public float Next()
        {
            lock (_lockObject)
            {
                var divider = (int) Math.Sqrt(_autoIncrementCurrentValue);

                if (divider == 0)
                {
                    divider = 1;
                }

                var ost0 = (_autoIncrementCurrentValue%divider);

                var result = 1f;

                if (ost0 > 0)
                {
                    result = (_autoIncrementCurrentValue%ost0)/(float) (ost0 + 1);
                }

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
                    buffer[i] = (byte) (_autoIncrementCurrentValue % 256);
                }
            }
        }
    }
}