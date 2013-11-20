using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator
{
    public class DodfCalculatorOld : IDodfCalculator
    {

        public DodfCalculatorOld(
            List<DataItem> fxwList)
        {
            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            _fxwList = fxwList;
        }

        public float[] CalculateDodf(
            int a)
        {
            this.FixForA(_fxwList, a);

            var part0 = new float[_fxwList[0].Input.Length];
            for (var b = 0; b < _fxwList.Count; b++)
            {
                if (_fxwList[a].OutputIndex == _fxwList[b].OutputIndex)
                {
                    var pab = this.GetPab(_fxwList, a, b);
                    var dab = this.GetDab(_fxwList, a, b);

                    for (var cc = 0; cc < dab.Length; cc++)
                    {
                        part0[cc] += pab * dab[cc];
                    }
                }
            }

            var part1 = new float[_fxwList[0].Input.Length];
            for (var b = 0; b < _fxwList.Count; b++)
            {
                if (_fxwList[a].OutputIndex == _fxwList[b].OutputIndex)
                {
                    var pab = this.GetPab(_fxwList, a, b);

                    var innerSum = new float[_fxwList[0].Input.Length];
                    for (var z = 0; z < _fxwList.Count; z++)
                    {
                        if (z != a)
                        {
                            var paz = this.GetPab(_fxwList, a, z);
                            var daz = this.GetDab(_fxwList, a, z);

                            for (var cc = 0; cc < daz.Length; cc++)
                            {
                                innerSum[cc] += paz * daz[cc];
                            }
                        }
                    }

                    for (var cc = 0; cc < innerSum.Length; cc++)
                    {
                        part1[cc] += pab * innerSum[cc];
                    }
                }
            }


            var part2 = new float[_fxwList[0].Input.Length];
            for (var l = 0; l < _fxwList.Count; l++)
            {
                if (_fxwList[a].OutputIndex == _fxwList[l].OutputIndex)
                {
                    this.FixForA(_fxwList, l);

                    var pla = this.GetPab(_fxwList, l, a);
                    var dla = this.GetDab(_fxwList, l, a);

                    for (var cc = 0; cc < dla.Length; cc++)
                    {
                        part2[cc] += pla * dla[cc];
                    }
                }
            }

            var part3 = new float[_fxwList[0].Input.Length];
            for (var l = 0; l < _fxwList.Count; l++)
            {
                if (l != a)
                {
                    this.FixForA(_fxwList, l);

                    var innerP = 0.0f;
                    for (var q = 0; q < _fxwList.Count; q++)
                    {
                        if (_fxwList[l].OutputIndex == _fxwList[q].OutputIndex)
                        {
                            var plq = this.GetPab(_fxwList, l, q);
                            innerP += plq;
                        }
                    }

                    var pla = this.GetPab(_fxwList, l, a);
                    var dla = this.GetDab(_fxwList, l, a);

                    for (var cc = 0; cc < dla.Length; cc++)
                    {
                        part3[cc] += innerP * pla * dla[cc];
                    }
                }
            }

            var dodf = new float[_fxwList[0].Input.Length];
            for (var cc = 0; cc < _fxwList[0].Input.Length; cc++)
            {
                dodf[cc] = -2f * (part0[cc] - part1[cc]) + 2f * (part2[cc] - part3[cc]);
            }

            return dodf;
        }

        private float _zn = 0.0f;
        private int _fixA = int.MinValue;
        private List<DataItem> _fxwList;

        private void FixForA(
            List<DataItem> fxwList,
            int a)
        {
            var zn = 0f;
            for (var z = 0; z < fxwList.Count; z++)
            {
                if (z != a)
                {
                    zn += GetExpDistanceab(fxwList, a, z);
                }
            }

            _zn = zn;
            _fixA = a;
        }

        private float GetPab(
            List<DataItem> fxwList,
            int a,
            int b
            )
        {
            if (_fixA != a)
            {
                throw new InvalidOperationException("_fixA != a");
            }

            if (a == b)
                return 0f;

            //числитель
            var ch = GetExpDistanceab(fxwList, a, b);

            return ch / _zn;
        }

        private float GetExpDistanceab(
            List<DataItem> fxwList,
            int a,
            int b)
        {
            var distance = GetDab(fxwList, a, b);

            return
                (float)Math.Exp(-(distance.Sum(j => j * j)));
        }

        private float[] GetDab(
            List<DataItem> fxwList,
            int a,
            int b)
        {
            var fxa = fxwList[a];
            var fxb = fxwList[b];

            var dab = new float[fxwList[a].Input.Length];
            for (var cc = 0; cc < fxwList[a].Input.Length; cc++)
            {
                dab[cc] = fxa.Input[cc] - fxb.Input[cc];
            }

            return
                dab;
        }
    }
}
