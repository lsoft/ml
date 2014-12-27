using System;

namespace MyNN.Common.Randomizer
{
    [Serializable]
    public class ConstRandomizer : IRandomizer
    {
        private readonly float _fvalue;
        private readonly int _ivalue;
        private readonly object _lockObject = new object();

        public ConstRandomizer()
        {
            
        }

        public ConstRandomizer(
            float fvalue,
            int ivalue
            )
        {
            _fvalue = fvalue;
            _ivalue = ivalue;
        }

        public int Next(int maxValue)
        {
            lock (_lockObject)
            {
                return
                    _ivalue%maxValue;
            }
        }

        public float Next()
        {
            lock (_lockObject)
            {
                return 
                    _fvalue;
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
                    buffer[i] = (byte)((_ivalue + i) % 256);
                }
            }
        }
    }
}