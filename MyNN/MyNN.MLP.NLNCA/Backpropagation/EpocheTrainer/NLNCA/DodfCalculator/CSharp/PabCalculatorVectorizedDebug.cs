using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.Data;
using MyNN.Common.OutputConsole;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.CSharp
{
    /// <summary>
    /// Correct but OBSOLETE implementation of p(a,b) calculator.
    /// </summary>
    public class PabCalculatorVectorizedDebug
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _inputLength;

        private readonly Dictionary<int, float[]> _expDistanceDict;
        private readonly List<float> _piList;

        private readonly List<float> _znList;

        public PabCalculatorVectorizedDebug(
            List<DataItem> fxwList)
        {
            #region validate

            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            #endregion

            _fxwList = fxwList;
            _inputLength = _fxwList[0].Input.Length;

            #region считаем distance dict

            var distanceDict = new ConcurrentDictionary<int, float[]>();

            var x0 = DateTime.Now;
            Parallel.For(0, _fxwList.Count, cc =>
                //for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                if ((_fxwList.Count - cc) > 0)
                {
                    var ourArray = new float[_fxwList.Count - cc];

                    ourArray[0] = 0f;
                    for (var dd = cc + 1; dd < _fxwList.Count; dd++)
                    {
                        var result = this.GetExpDistanceDab(cc, dd);

                        ourArray[dd - cc] = result;
                    }

                    distanceDict.TryAdd(cc, ourArray);
                }
            }
                );//Parallel.For

            _expDistanceDict = distanceDict.ToDictionary(j => j.Key, k => k.Value);

            var x1 = DateTime.Now;
            var diff1 = x1 - x0;
            ConsoleAmbientContext.Console.WriteLine("считаем distance dict = {0}", diff1);
            #endregion

            #region считаем знаменатель

            var x2 = DateTime.Now;
            _znList = new List<float>();

            for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                var zn = this.CalculateZnamenatelForA(cc);
                _znList.Add(zn);
            }
            var x3 = DateTime.Now;
            var diff2 = x3 - x2;
            ConsoleAmbientContext.Console.WriteLine("считаем знаменатель = {0}", diff2);

            #endregion

            #region считаем pi, pi * di

            var x4 = DateTime.Now;
            var internalDict = new Dictionary<int, List<int>>();
            for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                var key = _fxwList[cc].OutputIndex;

                if (!internalDict.ContainsKey(key))
                {
                    internalDict.Add(key, new List<int>());
                }

                internalDict[key].Add(cc);
            }
            var x5 = DateTime.Now;
            var diff3 = x5 - x4;
            ConsoleAmbientContext.Console.WriteLine("считаем pi, pi * di #1 = {0}", diff3);

            var x6 = DateTime.Now;
            _piList = new List<float>();
            //_pidiList = new List<float[]>();
            for (var l = 0; l < _fxwList.Count; l++)
            {
                var fxl = _fxwList[l];

                var pi = 0f;
                foreach (var q in internalDict[fxl.OutputIndex])
                {
                    var plq = this.GetPab(l, q);

                    pi += plq;
                }

                _piList.Add(pi);
            }
            var x7 = DateTime.Now;
            var diff4 = x7 - x6;
            ConsoleAmbientContext.Console.WriteLine("считаем pi, pi * di #2 = {0}", diff4);

            #endregion
        }

        private float CalculateZnamenatelForA(int a)
        {
            var zn = 0f;

            for (var z = 0; z < _fxwList.Count; z++)
            {
                if (z != a)
                {
                    zn += ExtractExpDistanceabFromDictionary(a, z);
                }
            }

            return zn;
        }


        public float GetPl(int l)
        {
            return
                _piList[l];
        }

        private float GetExpDistanceDab(
            int a,
            int b)
        {
            var fxa = _fxwList[a];
            var fxb = _fxwList[b];

            var sum = 0f;

            for (var cc = 0; cc < _inputLength; cc++)
            {
                var diff = fxa.Input[cc] - fxb.Input[cc];
                sum += diff * diff;
            }

#if DODF_DISABLE_EXP
            var result = -sum;
#else
            var result = (float)(Math.Exp(-sum));
#endif
            return result;
        }

        public void MADDab(
            ref float[] destination,
            float p,
            int a,
            int b)
        {
            var fxa = _fxwList[a];
            var fxb = _fxwList[b];

            #region dab = fxa - fxb;

            for (var cc = 0; cc < _inputLength; cc++)
            {
                var diff = fxa.Input[cc] - fxb.Input[cc];
                destination[cc] += diff * p;
            }

            #endregion
        }

        public void MAD2Dab(
            ref float[] part2,
            ref float[] part3,
            float pla,
            float f,
            int a,
            int b)
        {
            var fxa = _fxwList[a];
            var fxb = _fxwList[b];

            #region dab = fxa - fxb;

            for (var cc = 0; cc < _inputLength; cc++)
            {
                var diff = fxa.Input[cc] - fxb.Input[cc];
                part2[cc] += diff * pla;
                part3[cc] += diff * f;
            }

            #endregion
        }

        public float GetPab(
            int a,
            int b)
        {
            if (a == b)
            {
                return 0f;
            }

            //числитель
            var ch = ExtractExpDistanceabFromDictionary(a, b);

            var result = 0f;

            if (_znList[a] <= -float.Epsilon || _znList[a] >= float.Epsilon)
            {
                result = ch / _znList[a];
            }

            if (float.IsInfinity(result) || float.IsNaN(result) || float.IsNegativeInfinity(result) || float.IsPositiveInfinity(result))
            {
                throw new InvalidOperationException("float.IsInfinity(result) || float.IsNaN(result) || float.IsNegativeInfinity(result) || float.IsPositiveInfinity(result)");
            }


            return result;
        }

        private float ExtractExpDistanceabFromDictionary(
            int a,
            int b)
        {
            var distance = 0f;

            if (a == b)
            {
                throw new InvalidOperationException("a == b");
            }

            if (b > a)
            {
                distance = _expDistanceDict[a][b - a];
            }
            else if (b < a)
            {
                distance = _expDistanceDict[b][a - b];
            }

            return
                distance;
        }
    }
}