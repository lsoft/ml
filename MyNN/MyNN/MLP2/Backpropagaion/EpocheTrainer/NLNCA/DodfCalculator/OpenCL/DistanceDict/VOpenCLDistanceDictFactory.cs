using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public class VOpenCLDistanceDictFactory : IDistanceDictFactory
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

    int batchSize,
    int inputLength,
    int count)
{
    int defaultcc = get_global_id(0);
    int cc = defaultcc * batchSize;

    for(; cc < min((defaultcc + 1) * batchSize, count); cc++)
    {
        for (int dd = cc + 1; dd < count; dd++)
        {
//            //GetExpDistanceDab
//            float result = 0;
//            
//            for (int uu = 0; uu < inputLength; uu++)
//            {
//                float diff = fxwList[cc * inputLength + uu] - fxwList[dd * inputLength + uu];
//                result += diff * diff;
//            }

            //GetExpDistanceDab vectorized
            float16 result16 = 0;
            
            int uu = 0;
            for (; uu < inputLength - inputLength % 16; uu+=16)
            {
                int ccindex = cc * inputLength + uu;
                float16 fxwA = vload16(
                    ccindex / 16, 
                    fxwList + (ccindex %16));
    
                int ddindex = dd * inputLength + uu;
                float16 fxwB = vload16(
                    ddindex / 16, 
                    fxwList + (ddindex % 16));
    
                float16 diff16 = fxwA - fxwB;
                result16 += diff16 * diff16;
            }
    
            float result = 
                result16.s0 +
                result16.s1 +
                result16.s2 +
                result16.s3 +
                result16.s4 +
                result16.s5 +
                result16.s6 +
                result16.s7 +
                result16.s8 +
                result16.s9 +
                result16.sa +
                result16.sb +
                result16.sc +
                result16.sd +
                result16.se +
                result16.sf;
    
            for (; uu < inputLength; uu++)
            {
                float diff = fxwList[cc * inputLength + uu] - fxwList[dd * inputLength + uu];
                result += diff * diff;
            }

            distance[indexes[cc] + dd - cc] = result;//exp(-result);
        }
    }
}
",
        "DistanceKernel");

                var before = DateTime.Now;

                const int batchSize = 64;
                var kernelCount = fxwList.Count / batchSize;
                if (fxwList.Count % batchSize > 0)
                {
                    kernelCount++;
                }

                distanceKernel
                    .SetKernelArgMem(0, clProvider.FxwMem)
                    .SetKernelArgMem(1, clProvider.DistanceMem)
                    .SetKernelArgMem(2, clProvider.IndexMem)
                    .SetKernelArg(3, 4, batchSize)
                    .SetKernelArg(4, 4, inputLength)
                    .SetKernelArg(5, 4, fxwList.Count)
                    .EnqueueNDRangeKernel(kernelCount);

                // Make sure we're done with everything that's been requested before
                clProvider.QueueFinish();

                clProvider.DistanceMem.Read(BlockModeEnum.Blocking);

                var after = DateTime.Now;
                takenTime = (after - before);

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
