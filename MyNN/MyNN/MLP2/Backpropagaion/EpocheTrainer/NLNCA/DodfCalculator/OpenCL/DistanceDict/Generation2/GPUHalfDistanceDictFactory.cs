using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Accord.Statistics;
using MyNN.Data;
using OpenCL.Net.OpenCL.DeviceChooser;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation2
{
    /// <summary>
    /// 
    /// PS: To work correctly this kernel needs a disabled Tdr:
    /// http://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx
    /// Set HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers\TdrLevel = TdrLevelOff (0) - Detection disabled 
    /// and reboot. Otherwise, kernel will be aborted by Windows Video Driver Recovery Mechanism due to lack of response.
    /// </summary>
    public class GPUHalfDistanceDictFactory : IDistanceDictFactory
    {
        private const int DistanceMemElementCount = 1024 * 1024 * 10;
        private const float Threshold = 1e-15f;

        private readonly IDeviceChooser _deviceChooser;

        public GPUHalfDistanceDictFactory(
            IDeviceChooser deviceChooser)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            _deviceChooser = deviceChooser;
        }

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

            using (var clProvider = new DistanceDictHalfCLProvider(_deviceChooser, true, fxwList, DistanceMemElementCount))
            {
                var distanceKernel = clProvider.CreateKernel(@"
inline void WarpReductionToFirstElement(
    __local float *partialDotProduct)
{
#define WARP_SIZE 32

      // Thread local ID within a warp
      uint id = get_local_id(0) & (WARP_SIZE - 1); 

      // Each warp reduces 64 (default) consecutive elements
      float warpResult = 0.0f;
      if (get_local_id(0) < get_local_size(0)/2 )
      {
          volatile __local float* p = partialDotProduct + 2 * get_local_id(0) - id;
          p[0] += p[32];
          p[0] += p[16];
          p[0] += p[8];
          p[0] += p[4];
          p[0] += p[2];
          p[0] += p[1];
          warpResult = p[0];
      }

      // Synchronize to make sure each warp is done reading
      // partialDotProduct before it is overwritten in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // The first thread of each warp stores the result of the reduction
      // at the beginning of partialDotProduct
      if (id == 0)
         partialDotProduct[get_local_id(0) / WARP_SIZE] = warpResult;

      // Synchronize to make sure each warp is done writing to
      // partialDotProduct before it is read in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // Number of remaining elements after the first reduction
      uint size = get_local_size(0) / (2 * WARP_SIZE);

      // get_local_size(0) is less or equal to 512 on NVIDIA GPUs, so
      // only a single warp is needed for the following last reduction
      // step
      if (get_local_id(0) < size / 2)
      {
         volatile __local float* p = partialDotProduct + get_local_id(0);

         if (size >= 8)
            p[0] += p[4];
         if (size >= 4)
            p[0] += p[2];
         if (size >= 2)
            p[0] += p[1];
      }

}

inline int ObtainIndex(volatile __global int * indexArray)
{
    return
        atomic_inc(indexArray);
}

__kernel void DistanceKernel(
    const __global half * fxwList,
    __global float * distance,
    volatile __global int * indexArray,
    __local float * local_results,

    float threshold,
    int distanceMemLength,
    
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

            if(get_local_id(0) == 0)
            {
                float result = local_results[0];
                float exp_result = exp(-result);

                //exp_result = 123;
                if(exp_result >= threshold)
                {
                    int index = ObtainIndex(indexArray);
                    distance[index * 3 + 0] = cc;
                    distance[index * 3 + 1] = dd;
                    distance[index * 3 + 2] = exp_result;
                }
            }

            // Synchronize to make sure the first work-item is done with
            // reading partialDotProduct
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
",
                    "DistanceKernel");

                var before = DateTime.Now;

                const int szLocalSize = 128;
                int szGlobalSize = 16*clProvider.Parameters.NumComputeUnits*szLocalSize;

                distanceKernel
                    .SetKernelArgMem(0, clProvider.FxwMem)
                    .SetKernelArgMem(1, clProvider.DistanceMem)
                    .SetKernelArgMem(2, clProvider.IndexMem)
                    .SetKernelArgLocalMem(3, 4*szLocalSize)
                    .SetKernelArg(4, 4, Threshold)
                    .SetKernelArg(5, 4, DistanceMemElementCount)
                    .SetKernelArg(6, 4, inputLength)
                    .SetKernelArg(7, 4, fxwList.Count)
                    .EnqueueNDRangeKernel(
                        new int[]
                        {
                            szGlobalSize
                        }
                        , new int[]
                        {
                            szLocalSize
                        }
                    );

                // Make sure we're done with everything that's been requested before
                clProvider.QueueFinish();

                clProvider.DistanceMem.Read(BlockModeEnum.Blocking);
                clProvider.IndexMem.Read(BlockModeEnum.Blocking);

                var after = DateTime.Now;
                takenTime = (after - before);

                //колбасим в диктионари
                for (var cc = 0; cc < fxwList.Count; cc++)
                {
                    var iterSize = fxwList.Count - cc;

                    var array = new float[iterSize];
                    result.Add(cc, array);
                }

                for (var cc = 0; cc < DistanceMemElementCount; cc++)
                {
                    var distance = clProvider.DistanceMem.Array[cc * 3 + 2];
                    if (distance > 0)
                    {
                        var aIndex = (int)clProvider.DistanceMem.Array[cc * 3 + 0];
                        var bIndex = (int)clProvider.DistanceMem.Array[cc * 3 + 1];

                        result[aIndex][bIndex - aIndex] = distance;
                    }

                }

            }

            return result;
        }



    }
}
