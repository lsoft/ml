using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.Data;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.CSharp
{
    /// <summary>
    /// Correct but OBSOLETE implementation of p(a,b) calculator.
    /// </summary>
    public class PabCalculatorVectorized
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, float[]> _distanceDict;
        private readonly List<float> _piList;

        private readonly List<float> _znList;

        public PabCalculatorVectorized(
            List<DataItem> fxwList)
        {
            #region validate

            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            #endregion

            _fxwList = fxwList;
            _len = _fxwList[0].Input.Length;

            #region считаем distance dict

            var distanceDict = new ConcurrentDictionary<int, float[]>();

            Parallel.For(0, _fxwList.Count, cc =>
                //for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                if ((_fxwList.Count - cc) > 0)
                {
                    var ourArray = new float[_fxwList.Count - cc];

                    ourArray[0] = 0f;
                    for (var dd = cc + 1; dd < _fxwList.Count; dd++)
                    {
                        var dab = this.GetDab(cc, dd);

                        var result = dab.Sum(j => j * j);

                        ourArray[dd - cc] = result;
                    }

                    distanceDict.TryAdd(cc, ourArray);
                }
            }
                );//Parallel.For

            _distanceDict = distanceDict.ToDictionary(j => j.Key, k => k.Value);

            #endregion

            #region считаем знаменатель

            _znList = new List<float>();

            for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                var zn = this.CalculateZnamenatelForA(cc);
                _znList.Add(zn);
            }

            #endregion

            #region считаем pi, pi * di

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

            #endregion
        }

        private float CalculateZnamenatelForA(int a)
        {
            var zn = 0f;

            for (var z = 0; z < _fxwList.Count; z++)
            {
                if (z != a)
                {
                    zn += GetExpDistanceab(a, z);
                }
            }

            return zn;
        }


        public float GetPl(int l)
        {
            return
                _piList[l];
        }

        private float[] GetDab(
            int a,
            int b)
        {
            var fxa = _fxwList[a];
            var fxb = _fxwList[b];

            #region dab = fxa - fxb;

            var dab = new float[_len];
            for (var cc = 0; cc < _len; cc++)
            {
                dab[cc] = fxa.Input[cc] - fxb.Input[cc];
            }

            #endregion

            return
                dab;
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

            for (var cc = 0; cc < _len; cc++)
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

            for (var cc = 0; cc < _len; cc++)
            {
                var diff = fxa.Input[cc] - fxb.Input[cc];
                part2[cc] += diff * pla;
                part3[cc] += diff * f;
            }

            #endregion
        }

        public float GetPab(
            int a,
            int b
            )
        {
            if (a == b)
            {
                return 0f;
            }

            //числитель
            var ch = GetExpDistanceab(a, b);

            var result = 0f;

            if (_znList[a] <= -float.Epsilon || _znList[a] >= float.Epsilon)
            {
                result = ch / _znList[a];
            }

            return result;
        }

        private float GetExpDistanceab(
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
                distance = _distanceDict[a][b - a];
            }
            else if (b < a)
            {
                distance = _distanceDict[b][a - b];
            }

#if DODF_DISABLE_EXP
            var result = -distance;
#else
            var result = (float)Math.Exp(-distance);
#endif

            return result;
        }
    }
}