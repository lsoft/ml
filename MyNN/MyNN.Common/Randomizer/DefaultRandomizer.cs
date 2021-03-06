﻿using System;

namespace MyNN.Common.Randomizer
{
    [Serializable]
    public class DefaultRandomizer : IRandomizer
    {
        private readonly Random _random;
        private readonly object _lockObject = new object();

        public DefaultRandomizer(int rndSeed)
        {
            _random = new Random(rndSeed);
        }

        public int Next(int maxValue)
        {
            lock (_lockObject)
            {
                return
                    _random.Next(maxValue);
            }
        }

        public float Next()
        {
            lock (_lockObject)
            {
                return
                    (float) _random.NextDouble();
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
                _random.NextBytes(buffer);
            }
        }
    }
}
