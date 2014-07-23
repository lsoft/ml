using System;

namespace MyNNConsoleApp.DropConnectInference
{
    internal class MyNormal
    {
        private readonly double _median;
        private readonly double _sigma;
        private readonly Random _rnd;

        public MyNormal(double median, double sigma)
        {
            _median = median;
            _sigma = sigma;

            _rnd = new Random(DateTime.Now.Millisecond);
        }

        public double Sample()
        {
            //box muller
            var rnd1 = _rnd.NextDouble();
            var rnd2 = _rnd.NextDouble();
            var f = Math.Sqrt(- 2 * Math.Log(rnd1)) * Math.Cos(2 * Math.PI * rnd2);

            f = f*_sigma + _median;

            return f;
        }

    }
}