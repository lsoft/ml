using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator.OpenCL.DistanceDict;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator.OpenCL
{
    public class DodfCalculatorOpenCL : IDodfCalculator
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _len;

        private readonly Dictionary<int, List<int>> _fxwDict;
        private readonly PabCalculatorOpenCL _pabCalculator;

        public DodfCalculatorOpenCL(
            List<DataItem> fxwList,
            IDistanceDictFactory concreteDistanceDictFactory = null)
        {
            #region validate

            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }
            //concreteDistanceDictFactory allowed to be null

            #endregion

            _fxwList = fxwList;
            _len = _fxwList[0].Input.Length;
            _chooseDistanceDictFactory = concreteDistanceDictFactory;

            #region Заполняем _fxwDict

            var x0 = DateTime.Now;

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

            var x1 = DateTime.Now;
            var diff = x1 - x0;
            Console.WriteLine("Заполняем _fxwDict = {0}", diff);

            #endregion

            var distanceDictFactory = ChooseDistanceDictFactory(_fxwList);

            _pabCalculator = new PabCalculatorOpenCL(
                distanceDictFactory, 
                _fxwList);
        }

        /// <summary>
        /// Выбираем метод калькулирования (изза статика - один раз за запуск приложения)
        /// </summary>
        private static IDistanceDictFactory _chooseDistanceDictFactory;
        private IDistanceDictFactory ChooseDistanceDictFactory(List<DataItem> fxwList)
        {
            if (_chooseDistanceDictFactory == null)
            {
                var before = DateTime.Now;

                var csharp = new CSharpDistanceDictFactory();
                csharp.CreateDistanceDict(fxwList);

                var middle = DateTime.Now;

                var opencl = new OpenCLDistanceDictFactory();
                opencl.CreateDistanceDict(fxwList);

                var after = DateTime.Now;

                var diff0 = middle - before;
                var diff1 = after - middle;

                _chooseDistanceDictFactory =
                    diff1 < diff0
                        ? (IDistanceDictFactory) opencl
                        : (IDistanceDictFactory) csharp;
            }

            return
                _chooseDistanceDictFactory;

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
            }

            #region dodf = -2f * (part0 - part1) + 2f * (part2 - part3);

            var dodf = new float[_len];
            for (var cc = 0; cc < _len; cc++)
            {
                dodf[cc] = -2f * (part0[cc] - part1[cc]) + 2f * (part2[cc] - part3[cc]);
            }

            #endregion

            #region floating point checks

            if (dodf.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j)))
            {
                throw new InvalidOperationException("dodf.Any(j => float.IsInfinity(j) || float.IsNaN(j) || float.IsNegativeInfinity(j) || float.IsPositiveInfinity(j))");
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
