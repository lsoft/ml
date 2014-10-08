using System;
using System.Collections.Generic;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    /// <summary>
    /// dOdF distance container
    /// </summary>
    public class DodfDistanceContainer
    {
        private readonly Dictionary<int, Dictionary<int, float>> _dd;

        public int Count
        {
            get;
            private set;
        }

        public DodfDistanceContainer(int count)
        {
            _dd = new Dictionary<int, Dictionary<int, float>>();

            Count = count;
        }

        public void AddValue(int a, int b, float value)
        {
            if (a >= Count)
            {
                throw new ArgumentException("a >= Count");
            }
            if (b >= Count)
            {
                throw new ArgumentException("b >= Count");
            }

            if (!_dd.ContainsKey(a))
            {
                _dd.Add(a, new Dictionary<int, float>());
            }

            var i = _dd[a];

            i.Add(b, value);
        }

        public float GetDistance(int a, int b)
        {
            if (a >= Count)
            {
                throw new ArgumentException("a >= Count");
            }
            if (b >= Count)
            {
                throw new ArgumentException("b >= Count");
            }

            var ca = Math.Min(a, b);
            var cb = Math.Max(a, b);

            var result = 0f;

            if (_dd.ContainsKey(ca))
            {
                var i = _dd[ca];
                if (i.ContainsKey(cb))
                {
                    result = i[cb];
                }
            }

            return result;
        }
    }
}