using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1
{
    /// <summary>
    /// OBSOLETE and didn't debugged distance provider for dOdF algorithm uses C#.
    /// </summary>
    public class CSharpDistanceDictCalculator : IDistanceDictCalculator
    {
        public DodfDistanceContainer CalculateDistances(List<IDataItem> fxwList)
        {
            var inputLength = fxwList[0].Input.Length;

            var distanceDict = new ConcurrentDictionary<int, float[]>();

            Parallel.For(0, fxwList.Count, cc =>
            //for (var cc = 0; cc < fxwList.Count; cc++)
            {
                var ourArray = new float[fxwList.Count - cc];

                ourArray[0] = 0f;
                for (var dd = cc + 1; dd < fxwList.Count; dd++)
                {
                    var result = this.GetExpDistanceDab(
                        fxwList,
                        inputLength,
                        cc, 
                        dd);

                    ourArray[dd - cc] = result;
                }

                distanceDict.TryAdd(cc, ourArray);
            }
            ); //Parallel.For

            //колбасим в диктионари
            var resultD = new DodfDistanceContainer(fxwList.Count);

            for (var cc = 0; cc < fxwList.Count - 1; cc++)
            {
                for (var dd = cc + 1; dd < fxwList.Count; dd++)
                {
                    resultD.AddValue(cc, dd, distanceDict[cc][dd - cc]);
                }
            }

            return
                resultD; //distanceDict.ToDictionary(j => j.Key, k => k.Value);
        }


        private float GetExpDistanceDab(
            List<IDataItem> fxwList,
            int inputLength,
            int a,
            int b)
        {
            var fxa = fxwList[a];
            var fxb = fxwList[b];

            var sum = 0f;

            for (var cc = 0; cc < inputLength; cc++)
            {
                var diff = fxa.Input[cc] - fxb.Input[cc];
                sum += diff * diff;
            }

            float result;
            if (DoDfAmbientContext.DisableExponential)
            {
                result = -sum;
            }
            else
            {
                result = (float)(Math.Exp(-sum));
            }

            return result;
        }

    }
}
