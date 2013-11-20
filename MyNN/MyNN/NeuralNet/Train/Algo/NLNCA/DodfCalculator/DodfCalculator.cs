using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Data;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator
{
    public class DodfCalculator : IDodfCalculator
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, List<int>> _fxwDict;
        private readonly PabCalculator _pabCalculator;


        public DodfCalculator(
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

            _pabCalculator = new PabCalculator(
                _fxwList);
        }

        public float[] CalculateDodf(int a)
        {
            var aClass = _fxwList[a].OutputIndex;

            _pabCalculator.CalculateZnamenatelForA(a);

            var innerSum = new float[_len];
            for (var z = 0; z < _fxwList.Count; z++)
            {
                if (z != a)
                {
                    var paz = _pabCalculator.GetPab(a, z);
                    var daz = _pabCalculator.GetDab(a, z);

                    #region innerSum += daz * paz;

                    for (var cc = 0; cc < _len; cc++)
                    {
                        innerSum[cc] += daz[cc] * paz;
                    }

                    #endregion
                }

            }

            var part0 = new float[_len];
            var part1 = new float[_len];
            foreach (var b in _fxwDict[aClass])
            {
                var pab = _pabCalculator.GetPab(a, b);
                var dab = _pabCalculator.GetDab(a, b);

                #region part0 += dab * pab; и part1 += innerSum * pab;

                for (var cc = 0; cc < _len; cc++)
                {
                    part0[cc] += dab[cc] * pab;
                    part1[cc] += innerSum[cc] * pab;
                }

                #endregion
            }

            var part2 = new float[_len];
            var part3 = new float[_len];
            for (var l = 0; l < _fxwList.Count; l++)
            {
                _pabCalculator.CalculateZnamenatelForA(l);

                float pla = float.MinValue;
                float[] dla = null;

                if (aClass == _fxwList[l].OutputIndex)
                {
                    pla = _pabCalculator.GetPab(l, a);
                    dla = _pabCalculator.GetDab(l, a);

                    #region part2 += pla * dla;

                    for (var cc = 0; cc < dla.Length; cc++)
                    {
                        part2[cc] += pla * dla[cc];
                    }

                    #endregion
                }

                if (l != a)
                {
                    if (dla == null)
                    {
                        pla = _pabCalculator.GetPab(l, a);
                        dla = _pabCalculator.GetDab(l, a);
                    }

                    var innerP = 0.0f;
                    for (var q = 0; q < _fxwList.Count; q++)
                    {
                        if (_fxwList[l].OutputIndex == _fxwList[q].OutputIndex)
                        {
                            var plq = _pabCalculator.GetPab(l, q);
                            innerP += plq;
                        }
                    }

                    #region part3 += innerP * pla * dla;

                    for (var cc = 0; cc < dla.Length; cc++)
                    {
                        part3[cc] += innerP * pla * dla[cc];
                    }

                    #endregion
                }
            }

            #region dodf = -2f * (part0 - part1) + 2f * (part2 - part3);

            var dodf = new float[_len];
            for (var cc = 0; cc < _len; cc++)
            {
                dodf[cc] = -2f * (part0[cc] - part1[cc]) + 2f * (part2[cc] - part3[cc]);
            }

            #endregion

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
                        _pabCalculator.CalculateZnamenatelForA(cc);

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

    public class PabCalculator
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _len;

        private float _znamenatel = 0.0f;
        private int _fixA = int.MinValue;

        private readonly Dictionary<int, Dictionary<int, float>> _distanceDict;

        public PabCalculator(
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

            _distanceDict = new Dictionary<int, Dictionary<int, float>>();

            for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                var ourDict = new Dictionary<int, float>();

                for (var dd = 0; dd < _fxwList.Count; dd++)
                {
                    var result = 0f;

                    if (dd > cc)
                    {
                        var dab = this.GetDab(cc, dd);

                        result = dab.Sum(j => j * j);
                    }

                    ourDict.Add(dd, result);
                }

                _distanceDict.Add(cc, ourDict);
            }
        }

        public float[] GetDab(
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

        public void CalculateZnamenatelForA(int a)
        {
            var zn = 0f;

            for (var z = 0; z < _fxwList.Count; z++)
            {
                if (z != a)
                {
                    zn += GetExpDistanceab(a, z);
                }
            }

            _znamenatel = zn;
            _fixA = a;
        }

        public float GetPab(
            int a,
            int b
            )
        {
            #region validate

            if (_fixA != a)
            {
                throw new InvalidOperationException("_fixA != a");
            }

            #endregion

            if (a == b)
            {
                return 0f;
            }

            //числитель
            var ch = GetExpDistanceab(a, b);

            return ch / _znamenatel;
        }

        public float GetExpDistanceab(
            int a,
            int b)
        {
            var distance = 0f;

            if (b > a)
            {
                distance = _distanceDict[a][b];
            }
            else if (b < a)
            {
                distance = _distanceDict[b][a];
            }

            return
                (float)Math.Exp(-distance);
        }
    }
}
