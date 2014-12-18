using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.CSharp
{
    /// <summary>
    /// Correct but OBSOLETE implementation of p(a,b) calculator.
    /// </summary>
    public class PabCalculator
    {
        private readonly new List<IDataItem> _fxwList;
        private readonly int _len;

        private float _znamenatel = 0.0f;
        private int _fixA = int.MinValue;

        private readonly Dictionary<int, Dictionary<int, float>> _distanceDict;

        public PabCalculator(
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

#if DODF_DISABLE_EXP
            var result = -distance;
#else
            var result = (float)Math.Exp(-distance);
#endif

            return result;
        }
    }
}