//#define DODF_DEBUG_CHECKS //раскоментировать, если нужна проверка на NaN, +-Infinity и т.п.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL
{
    /// <summary>
    /// OpenCL implementation of dOdF calculator.
    /// </summary>
    public class DodfCalculatorOpenCL : IDodfCalculator
    {
        private readonly List<IDataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, List<int>> _fxwDict;
        private readonly PabCalculatorOpenCL _pabCalculator;


        public DodfCalculatorOpenCL(
            List<IDataItem> fxwList,
            IDistanceDictCalculator distanceDictCalculator)
        {
            #region validate

            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }
            if (distanceDictCalculator == null)
            {
                throw new ArgumentNullException("distanceDictCalculator");
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

            _pabCalculator = new PabCalculatorOpenCL(
                distanceDictCalculator, 
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

#if DODF_DEBUG_CHECKS
                #region floating point checks

                if (part2.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
                {
                    throw new InvalidOperationException("part2.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
                }
                if (part3.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
                {
                    throw new InvalidOperationException("part3.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
                }

                #endregion
#endif
            }

            #region dodf = -2f * (part0 - part1) + 2f * (part2 - part3);

            var dodf = new float[_len];
            for (var cc = 0; cc < _len; cc++)
            {
                dodf[cc] = -2f * (part0[cc] - part1[cc]) + 2f * (part2[cc] - part3[cc]);
            }

            #endregion

#if DODF_DEBUG_CHECKS
            #region floating point checks

            if (dodf.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
            {
                throw new InvalidOperationException("dodf.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
            }

            #endregion
#endif

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
}
