using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL
{
    /// <summary>
    /// OpenCL implementation of p(a,b) calculator.
    /// </summary>
    public class PabCalculatorOpenCL 
    {
        private readonly List<IDataItem> _fxwList;
        private readonly int _inputLength;

        private readonly DodfDistanceContainer _expDistanceDict;
        private readonly List<float> _piList;

        private readonly List<float> _znList;

        public PabCalculatorOpenCL(
            IDistanceDictCalculator createDistanceDictCalculator,
            List<IDataItem> fxwList)
        {
            if (createDistanceDictCalculator == null)
            {
                throw new ArgumentNullException("createDistanceDictCalculator");
            }
            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            _fxwList = fxwList;
            _inputLength = _fxwList[0].Input.Length;

            #region считаем distance dict

            _expDistanceDict = createDistanceDictCalculator.CalculateDistances(_fxwList);

            #endregion

            #region считаем знаменатель

            _znList = new float[_fxwList.Count].ToList();

            Parallel.For(0, _fxwList.Count, cc =>
            //for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                var zn = this.CalculateZnamenatelForA(cc);
                _znList[cc] = zn;
            }
            );//Parallel.For

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
        
        public float GetPl(int l)
        {
            return
                _piList[l];
        }

        public void MADDab(
            ref float[] destination,
            float p,
            int a,
            int b)
        {
            var fxa = _fxwList[a];
            var fxb = _fxwList[b];

            #region dab += (fxa - fxb) * p;

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

#if PAB_DEBUG_CHECKS
            if (float.IsInfinity(result) || float.IsNaN(result) || float.IsNegativeInfinity(result) || float.IsPositiveInfinity(result))
            {
                throw new InvalidOperationException("float.IsInfinity(result) || float.IsNaN(result) || float.IsNegativeInfinity(result) || float.IsPositiveInfinity(result)");
            }
#endif

            return result;
        }

        #region private methods

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

        private float ExtractExpDistanceabFromDictionary(
            int a,
            int b)
        {
            var distance = 0f;

            if (a == b)
            {
                throw new InvalidOperationException("a == b");
            }

            distance = _expDistanceDict.GetDistance(a, b);

            return
                distance;
        }

        #endregion
    }
}