using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Single;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator
{
    //public class DodfCalculatorMathNet
    //{
    //    private readonly int[] _classList;

    //    public DodfCalculatorMathNet(
    //        int[] classList)
    //    {
    //        if (classList == null)
    //        {
    //            throw new ArgumentNullException("classList");
    //        }

    //        _classList = classList;
    //    }

    //    public DenseVector CalculateDodf(
    //        int a,
    //        List<Pair<int, float[]>> fxwList
    //        )
    //    {
    //        this.FixForA(fxwList, a);

    //        var part0 = new DenseVector(fxwList[0].Second.Length);
    //        for (var b = 0; b < fxwList.Count; b++)
    //        {
    //            if (_classList[a] == _classList[b])
    //            {
    //                var pab = this.GetPab(fxwList, a, b);
    //                var dab = this.GetDab(fxwList, a, b);

    //                part0 += pab * dab;
    //            }
    //        }

    //        var part1 = new DenseVector(fxwList[0].Second.Length);
    //        for (var b = 0; b < fxwList.Count; b++)
    //        {
    //            if (_classList[a] == _classList[b])
    //            {
    //                var pab = this.GetPab(fxwList, a, b);

    //                var innerSum = new DenseVector(fxwList[0].Second.Length);
    //                for (var z = 0; z < fxwList.Count; z++)
    //                {
    //                    if (z != a)
    //                    {
    //                        var paz = this.GetPab(fxwList, a, z);
    //                        var daz = this.GetDab(fxwList, a, z);

    //                        innerSum += paz * daz;
    //                    }
    //                }

    //                part1 += pab * innerSum;
    //            }
    //        }


    //        var part2 = new DenseVector(fxwList[0].Second.Length);
    //        for (var l = 0; l < fxwList.Count; l++)
    //        {
    //            if (_classList[a] == _classList[l])
    //            {
    //                this.FixForA(fxwList, l);

    //                var pla = this.GetPab(fxwList, l, a);
    //                var dla = this.GetDab(fxwList, l, a);

    //                part2 += pla * dla;
    //            }
    //        }

    //        var part3 = new DenseVector(fxwList[0].Second.Length);
    //        for (var l = 0; l < fxwList.Count; l++)
    //        {
    //            if (l != a)
    //            {
    //                this.FixForA(fxwList, l);

    //                var innerP = 0.0f;
    //                for (var q = 0; q < fxwList.Count; q++)
    //                {
    //                    if (_classList[l] == _classList[q])
    //                    {
    //                        var plq = this.GetPab(fxwList, l, q);
    //                        innerP += plq;
    //                    }
    //                }

    //                var pla = this.GetPab(fxwList, l, a);
    //                var dla = this.GetDab(fxwList, l, a);

    //                part3 += innerP * pla * dla;
    //            }
    //        }

    //        var dodf = -2f * (part0 - part1) + 2f * (part2 - part3);

    //        return dodf;
    //    }

    //    private float _zn = 0.0f;
    //    private int _fixA = int.MinValue;

    //    private void FixForA(
    //        List<Pair<int, float[]>> fxwList,
    //        int a)
    //    {
    //        var zn = 0f;
    //        for (var z = 0; z < fxwList.Count; z++)
    //        {
    //            if (z != a)
    //            {
    //                zn += GetExpDistanceab(fxwList, a, z);
    //            }
    //        }

    //        _zn = zn;
    //        _fixA = a;
    //    }

    //    private float GetPab(
    //        List<Pair<int, float[]>> fxwList,
    //        int a,
    //        int b
    //        )
    //    {
    //        if (_fixA != a)
    //        {
    //            throw new InvalidOperationException("_fixA != a");
    //        }

    //        if (a == b)
    //            return 0f;

    //        //числитель
    //        var ch = GetExpDistanceab(fxwList, a, b);

    //        return ch / _zn;
    //    }

    //    private float GetExpDistanceab(
    //        List<Pair<int, float[]>> fxwList,
    //        int a,
    //        int b)
    //    {
    //        var distance = GetDab(fxwList, a, b);

    //        return
    //            (float)Math.Exp(-(distance * distance));
    //    }

    //    private DenseVector GetDab(
    //        List<Pair<int, float[]>> fxwList,
    //        int a,
    //        int b)
    //    {
    //        var fxa = new DenseVector(fxwList[a].Second);
    //        var fxb = new DenseVector(fxwList[b].Second);
    //        var dab = fxa - fxb;

    //        return
    //            dab;
    //    }
    //}
}
