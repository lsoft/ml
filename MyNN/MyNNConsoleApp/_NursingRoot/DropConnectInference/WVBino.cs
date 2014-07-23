using System;

namespace MyNNConsoleApp.DropConnectInference
{
    internal class WVBino
    {
        private readonly int _n;
        private readonly float[] _wv;
        private readonly float _p;
        private Random _rnd;

        public WVBino(float p, int n, float[] wv)
        {
            if (wv == null || wv.Length != n)
            {
                throw new ArgumentNullException("wv");
            }

            _n = n;
            _wv = wv;
            _p = p;

            _rnd = new Random(DateTime.Now.Millisecond);
        }

        public double GetU()
        {
            var u = 0.0;

            for (var index = 0; index < _n; index++)
            {
                var isone = _rnd.NextDouble() < _p;

                if (isone)
                {
                    u += _wv[index];
                }
            }

            return u;
        }

        public double Probability(int P)
        {
            var rnd = new Random(DateTime.Now.Millisecond);

            const int totalExp = 100000;

            var hitExp = 0;
            for (var exp = 0; exp < totalExp; exp++)
            {
                var total = 0;
                for (var r = 0; r < _n; r++)
                {
                    var isone = rnd.NextDouble() < _p;

                    if (isone)
                    {
                        total++;
                    }
                }

                if (total == P)
                {
                    hitExp++;
                }
            }

            return
                hitExp / (double)totalExp;
        }
    }
}