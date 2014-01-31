﻿using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public class OpenCLDistanceDictFactory : IDistanceDictFactory
    {
        public DodfDictionary CreateDistanceDict(List<DataItem> fxwList)
        {
            TimeSpan takenTime;

            return
                CreateDistanceDict(fxwList, out takenTime);
        }

        public DodfDictionary CreateDistanceDict(List<DataItem> fxwList, out TimeSpan takenTime)
        {
            var result = new DodfDictionary(fxwList.Count);

            var inputLength = fxwList[0].Input.Length;

            using (var clProvider = new DistanceDictCLProvider(fxwList))
            {
                var kernelText = @"
{DODF_DISABLE_EXP_DEFINE_CLAUSE}

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
            
#ifdef DODF_DISABLE_EXP
        float write_result = -result;
#else
        float write_result = exp(-result);
#endif

        distance[indexes[cc] + dd - cc] = write_result;
    }
}
";

#if DODF_DISABLE_EXP
                kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", "#define DODF_DISABLE_EXP");
#else
                kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", string.Empty);
#endif

                var distanceKernel = clProvider.CreateKernel(kernelText, "DistanceKernel");

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

                clProvider.DistanceMem.Read(BlockModeEnum.Blocking);

                var after = DateTime.Now;
                takenTime = (after - before);

                //колбасим в диктионари
                var pointer = 0;
                for (var cc = 0; cc < fxwList.Count - 1; cc++)
                {
                    pointer++; //correct for dd start value (not cc, but cc + 1)

                    for (var dd = cc + 1; dd < fxwList.Count; dd++)
                    {
                        result.AddValue(cc, dd, clProvider.DistanceMem.Array[pointer]);

                        pointer++;
                    }
                }
            }

            return result;
        }



    }
}
