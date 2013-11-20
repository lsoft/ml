
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Data;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator
{
    public class DodfCalculatorVectorizedDebug : IDodfCalculator
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, List<int>> _fxwDict;
        private readonly PabCalculatorVectorizedDebug _pabCalculator;


        public DodfCalculatorVectorizedDebug(
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

            #region Заполняем _fxwDict

            _fxwDict = new Dictionary<int, List<int>>();

            var classDistinctList = (from x in _fxwList select x.OutputIndex).Distinct();
            foreach (var cId in classDistinctList)
            {
                _fxwDict.Add(cId, new List<int>());
            }

            for (var a = 0; a < _fxwList.Count; a++)
            {
                _fxwDict[_fxwList[a].OutputIndex].Add(a);
            }

            #endregion

            _pabCalculator = new PabCalculatorVectorizedDebug(
                _fxwList);
        }

        public float[] CalculateDodf(int a)
        {
            var aClass = _fxwList[a].OutputIndex;

            var part0 = new float[_len];
            var part1 = new float[_len];

            var innerSum = new float[_len];
            for (var z = 0; z < _fxwList.Count; z++)
            {
                if (z != a)
                {
                    var paz = _pabCalculator.GetPab(a, z);
                    _pabCalculator.MADDab(ref innerSum, paz, a, z);
                }

            }

            foreach (var b in _fxwDict[aClass])
            {
                var pab = _pabCalculator.GetPab(a, b);
                _pabCalculator.MADDab(ref part0, pab, a, b);

                #region part0 += dab * pab; и part1 += innerSum * pab;

                for (var cc = 0; cc < _len; cc++)
                {
                    part1[cc] += innerSum[cc] * pab;
                }

                #endregion
            }

            var part2 = new float[_len];
            var part3 = new float[_len];
            for (var l = 0; l < _fxwList.Count; l++)
            {
                var la = (l != a);
                var domain = aClass == _fxwList[l].OutputIndex;

                var pla = _pabCalculator.GetPab(l, a);

                float innerP = 0f;

                if (la)
                {
                    innerP = _pabCalculator.GetPl(l);
                }

                float part2Coef = 1f;

                if (!domain)
                {
                    part2Coef = 0f;
                }

                _pabCalculator.MAD2Dab(
                    ref part2,
                    ref part3,
                    pla * part2Coef,
                    pla * innerP,
                    l,
                    a);

                if (part2.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
                {
                    throw new InvalidOperationException("part2.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
                }
                if (part3.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
                {
                    throw new InvalidOperationException("part3.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
                }
            }

            #region dodf = -2f * (part0 - part1) + 2f * (part2 - part3);

            var dodf = new float[_len];
            for (var cc = 0; cc < _len; cc++)
            {
                dodf[cc] = -2f * (part0[cc] - part1[cc]) + 2f * (part2[cc] - part3[cc]);
            }

            #endregion

            if (dodf.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
            {
                throw new InvalidOperationException("dodf.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
            }

            return dodf;
        }

        public void DumpPi(string csvFile, bool withHeader = false)
        {
            var headerResult = "";
            var stringResult = "";
            foreach (var key in _fxwDict.Keys.OrderBy(j => j))
            {
                for (var cc = 0; cc < _fxwList.Count; cc++)
                {
                    if (_fxwList[cc].OutputIndex == key)
                    {
                        var pi = 0f;

                        for (var dd = 0; dd < _fxwList.Count; dd++)
                        {
                            if (_fxwList[cc].OutputIndex == _fxwList[dd].OutputIndex)
                            {
                                var pccdd = _pabCalculator.GetPab(cc, dd);
                                pi += pccdd;
                            }
                        }

                        headerResult += _fxwList[cc].OutputIndex.ToString() + ";";
                        stringResult += pi.ToString() + ";";
                    }
                }
            }

            if (withHeader)
            {
                File.AppendAllText(csvFile, headerResult + "\r\n");
            }
            File.AppendAllText(csvFile, stringResult + "\r\n");

        }
    }

    public class PabCalculatorVectorizedDebug
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, float[]> _distanceDict;
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

            if (float.IsInfinity(result) || float.IsNaN(result) || float.IsNegativeInfinity(result) || float.IsPositiveInfinity(result))
            {
                throw new InvalidOperationException("float.IsInfinity(result) || float.IsNaN(result) || float.IsNegativeInfinity(result) || float.IsPositiveInfinity(result)");
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

            return
                (float)Math.Exp(-distance);
        }
    }
}
