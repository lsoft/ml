using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public class OpenCLDistanceDictFactory : IDistanceDictFactory
    {
        public Dictionary<int, float[]> CreateDistanceDict(List<DataItem> fxwList)
        {
            TimeSpan takenTime;

            return
                CreateDistanceDict(fxwList, out takenTime);
        }

        public Dictionary<int, float[]> CreateDistanceDict(List<DataItem> fxwList, out TimeSpan takenTime)
        {
            var result = new Dictionary<int, float[]>();

            var inputLength = fxwList[0].Input.Length;

            using (var clProvider = new DistanceDictCLProvider(fxwList))
            {
                var distanceKernel = clProvider.CreateKernel(@"
__kernel void DistanceKernel(
    __global float * fxwList,
    __global float * distance,
    __global int * indexes,
            
    int inputLength,
    int count)
{
    int cc = get_global_id(0);

    for (int dd = cc + 1; dd < count; dd++)
    {
        //GetExpDistanceDab
        float result = 0;
            
        for (int uu = 0; uu < inputLength; uu++)
        {
            float diff = fxwList[cc * inputLength + uu] - fxwList[dd * inputLength + uu];
            result += diff * diff;
        }
            
        distance[indexes[cc] + dd - cc] = exp(-result);
    }
}
",
        "DistanceKernel");

                var before = DateTime.Now;

                distanceKernel
                    .SetKernelArgMem(0, clProvider.FxwMem)
                    .SetKernelArgMem(1, clProvider.DistanceMem)
                    .SetKernelArgMem(2, clProvider.IndexMem)
                    .SetKernelArg(3, 4, inputLength)
                    .SetKernelArg(4, 4, fxwList.Count)
                    .EnqueueNDRangeKernel(fxwList.Count);

                // Make sure we're done with everything that's been requested before
                clProvider.QueueFinish();

                var after = DateTime.Now;
                takenTime = (after - before);

                clProvider.DistanceMem.Read(BlockModeEnum.Blocking);

                //колбасим в диктионари
                var pointer = 0;
                for (var cc = 0; cc < fxwList.Count; cc++)
                {
                    var iterSize = fxwList.Count - cc;

                    var array = new float[iterSize];
                    Array.Copy(
                        clProvider.DistanceMem.Array,
                        pointer,
                        array,
                        0,
                        iterSize);

                    result.Add(cc, array);

                    pointer += iterSize;
                }
            }

            return result;
        }



    }
}
