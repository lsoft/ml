using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.CSharp
{
    /// <summary>
    /// Correct but OBSOLETE implementation of dOdF calculator.
    /// </summary>
    public class DodfCalculator : IDodfCalculator
    {
        private readonly new List<IDataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, List<int>> _fxwDict;
        private readonly PabCalculator _pabCalculator;


        public DodfCalculator(
            List<IDataItem> fxwList)
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
}
