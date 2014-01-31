using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;

using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    /// <summary>
    /// 
    /// PS: To work correctly this kernel needs a disabled Tdr:
    /// http://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx
    /// Set HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers\TdrLevel = TdrLevelOff (0) - Detection disabled 
    /// Otherwise, kernel will be aborted by Windows Video Driver Recovery Mechanism due to lack of response.
    /// </summary>
    public class GPUNaiveDistanceDictFactory : IDistanceDictFactory
    {
        private readonly IDeviceChooser _deviceChooser;

        public GPUNaiveDistanceDictFactory(
            IDeviceChooser deviceChooser)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            _deviceChooser = deviceChooser;
        }

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

            using (var clProvider = new DistanceDictCLProvider(_deviceChooser, true, fxwList))
            {
                var kernelText = @"
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

{DODF_DISABLE_EXP_DEFINE_CLAUSE}

__kernel void DistanceKernel(
    const __global float * fxwList,
    __global float * distance,
    const __global int * indexes,
    __local float * local_results,

    int inputLength,
    int count)
{
    for (uint cc = get_group_id(0); cc < count; cc += get_num_groups(0))
    {
        __global float * aElement = fxwList + cc * inputLength;

        for (int dd = cc + 1; dd < count; dd++)
        {
            //GetExpDistanceDab
            __global float * bElement = fxwList + dd * inputLength;
            
            float local_result = 0;
            for (int uu = get_local_id(0); uu < inputLength; uu += get_local_size(0))
            {
                float diff = aElement[uu] - bElement[uu];
                local_result += diff * diff;
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

#if DODF_DISABLE_EXP
                kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", "#define DODF_DISABLE_EXP");
#else
                kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", string.Empty);
#endif

                var distanceKernel = clProvider.CreateKernel(kernelText, "DistanceKernel");

                var before = DateTime.Now;

                const int szLocalSize = 128;
                int szGlobalSize = 16 * clProvider.Parameters.NumComputeUnits * szLocalSize;

                distanceKernel
                    .SetKernelArgMem(0, clProvider.FxwMem)
                    .SetKernelArgMem(1, clProvider.DistanceMem)
                    .SetKernelArgMem(2, clProvider.IndexMem)
                    .SetKernelArgLocalMem(3, 4 * szLocalSize)
                    .SetKernelArg(4, 4, inputLength)
                    .SetKernelArg(5, 4, fxwList.Count)
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
