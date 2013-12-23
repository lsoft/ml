using System;

namespace MyNNConsoleApp.DropConnectInference
{
    internal class MyBino
    {
        private readonly int _n;
        private readonly float _p;

        public MyBino(float p, int n)
        {
            _n = n;
            _p = p;
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
                hitExp/(double) totalExp;
        }
    }
}