using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Data;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public class CSharpDistanceDictFactory : IDistanceDictFactory
    {
        public Dictionary<int, float[]> CreateDistanceDict(List<DataItem> fxwList)
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

            return
                distanceDict.ToDictionary(j => j.Key, k => k.Value);
        }


        private float GetExpDistanceDab(
            List<DataItem> fxwList,
            int inputLength,
            int a,
            int b)
        {
            var fxa = fxwList[a];
            var fxb = fxwList[b];

            var result = 0f;

            for (var cc = 0; cc < inputLength; cc++)
            {
                var diff = fxa.Input[cc] - fxb.Input[cc];
                result += diff * diff;
            }

            return
                (float)(Math.Exp(-result));
        }

    }
}
