﻿using System;
using System.Collections.Generic;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation2
{
    /// <summary>
    /// Correct but OBSOLETE implementations of distance provider for dOdF algorithm that enables GPU-OpenCL with HALF input representation.
    /// PS: To work correctly this kernel needs a disabled Tdr:
    /// http://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx
    /// Set HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers\TdrLevel = TdrLevelOff (0) - Detection disabled 
    /// and reboot. Otherwise, kernel will be aborted by Windows Video Driver Recovery Mechanism due to lack of response.
    /// </summary>
    public class GpuHalfNaiveDistanceDictCalculator : IDistanceDictCalculator
    {
        private readonly IDeviceChooser _deviceChooser;

        public GpuHalfNaiveDistanceDictCalculator(
            IDeviceChooser deviceChooser)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            _deviceChooser = deviceChooser;
        }

        public DodfDistanceContainer CalculateDistances(List<IDataItem> fxwList)
        {
            TimeSpan takenTime;

            return
                CreateDistanceDict(fxwList, out takenTime);
        }

        public DodfDistanceContainer CreateDistanceDict(List<IDataItem> fxwList, out TimeSpan takenTime)
        {
            var result = new DodfDistanceContainer(fxwList.Count);

            var inputLength = fxwList[0].Input.Length;

            using (var clProvider = new OpenCLDistanceDictHalfProvider(_deviceChooser, true, fxwList))
            {
                var kernelText = @"
{DODF_DISABLE_EXP_DEFINE_CLAUSE}

__kernel void DistanceKernel(
    const __global half * fxwList,
    __global float * distance,
    const __global int * indexes,
    __local float * local_results,

    int inputLength,
    int count)
{
    for (uint cc = get_group_id(0); cc < count; cc += get_num_groups(0))
    {
        //__global half * aElement = fxwList + cc * inputLength;
        uint aIndex = cc * inputLength;
        uint aShift = aIndex % 2;

        for (int dd = cc + 1; dd < count; dd++)
        {
            //------------------------ GetExpDistanceDab ------------------------------

//            __global half * bElement = fxwList + dd * inputLength ;
//            
//            float local_result = 0;
//            for (int uu = get_local_id(0); uu < inputLength; uu += get_local_size(0))
//            {
//                float afloat = vload_half(uu, aElement);
//                float bfloat = vload_half(uu, bElement);
//                float diff = afloat - bfloat;
//                local_result += diff * diff;
//            }

            uint bIndex = dd * inputLength;
            uint bShift = bIndex % 2;

            uint ostatok = inputLength % 2;

            float local_result = 0;
            for (int uu = get_local_id(0) * 2; uu < inputLength - ostatok; uu += get_local_size(0) * 2)
            {
                float2 afloat2 = vload_half2((aIndex + uu) / 2, fxwList + aShift);
                float2 bfloat2 = vload_half2((bIndex + uu) / 2, fxwList + bShift);
                float2 diff = afloat2 - bfloat2;
                local_result += diff.x * diff.x + diff.y * diff.y;
            }

            if(get_local_id(0) == 0)
            {
                if(ostatok > 0)
                {
                    float afloat = vload_half((cc + 1) * inputLength - 1, fxwList);
                    float bfloat = vload_half((dd + 1) * inputLength - 1, fxwList);
                    float diff = afloat - bfloat;
                    local_result += diff * diff;
                }
            }

            local_results[get_local_id(0)] = local_result;
            barrier(CLK_LOCAL_MEM_FENCE);

            WarpReductionToFirstElement(local_results);
            barrier(CLK_LOCAL_MEM_FENCE);
            float result = local_results[0];

            if(get_local_id(0) == 0)
            {
#ifdef DODF_DISABLE_EXP
                float write_result = -result;
#else
                float write_result = exp(-result);
#endif

                distance[indexes[cc] + dd - cc] = write_result;
            }

            // Synchronize to make sure the first work-item is done with
            // reading partialDotProduct
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
";

                if (DoDfAmbientContext.DisableExponential)
                {
                    kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", "#define DODF_DISABLE_EXP");
                }
                else
                {
                    kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", string.Empty);
                }


                var distanceKernel = clProvider.CreateKernel(kernelText, "DistanceKernel");

                var before = DateTime.Now;

                const uint szLocalSize = 128;
                uint szGlobalSize = 16 * clProvider.Parameters.NumComputeUnits * szLocalSize;

                distanceKernel
                    .SetKernelArgMem(0, clProvider.FxwMem)
                    .SetKernelArgMem(1, clProvider.DistanceMem)
                    .SetKernelArgMem(2, clProvider.IndexMem)
                    .SetKernelArgLocalMem(3, 4 * szLocalSize)
                    .SetKernelArg(4, 4, inputLength)
                    .SetKernelArg(5, 4, fxwList.Count)
                    .EnqueueNDRangeKernel(
                        new []
                        {
                            szGlobalSize
                        }
                        , new []
                        {
                            szLocalSize
                        }
                        );

                // Make sure we're done with everything that's been requested before
                clProvider.QueueFinish();

                clProvider.DistanceMem.Read(BlockModeEnum.Blocking);

                var after = DateTime.Now;
                takenTime = (after - before);

                //var countGreatherThan1eM20 = clProvider.DistanceMem.Array.Sum(j => j >= 1e-15 ? 1 : 0);
                //var min = clProvider.DistanceMem.Array.Min();
                //var avg = clProvider.DistanceMem.Array.Average();
                //var max = clProvider.DistanceMem.Array.Max();
                //Console.WriteLine(
                //    "min = {0}, avg = {1}, max = {2}, (count > 1e-15) = {3}% ({4}/{5})", 
                //    min, 
                //    avg, 
                //    max, 
                //    ((int)(countGreatherThan1eM20 * 10000f / (float)clProvider.DistanceMem.Array.Length)) / 100f,
                //    countGreatherThan1eM20,
                //    clProvider.DistanceMem.Array.Length);

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
