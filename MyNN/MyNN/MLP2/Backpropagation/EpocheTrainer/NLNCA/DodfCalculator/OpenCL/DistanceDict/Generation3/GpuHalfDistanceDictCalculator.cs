using System;
using System.Collections.Generic;
using MyNN.Data;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3
{
    /// <summary>
    /// Correct implementations of distance provider for dOdF algorithm that enables GPU-OpenCL with HALF input representation.
    /// This implementation contains optimizations in memory consumption in distance table and introduce threshold in distance metric.
    /// PS: To work correctly this kernel needs a disabled Tdr:
    /// http://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx
    /// Set HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers\TdrLevel = TdrLevelOff (0) - Detection disabled 
    /// and reboot. Otherwise, kernel will be aborted by Windows Video Driver Recovery Mechanism due to lack of response.
    /// </summary>
    public class GpuHalfDistanceDictCalculator : IDistanceDictCalculator
    {
        /// <summary>
        /// Distance Mem object length in element (in bytes it will be bigger 12 times (4 byte per float * 3 float per element)
        /// </summary>
        private const int DistanceMemElementCount = 1024 * 1024 * 10;

        /// <summary>
        /// Threshold for distance.
        /// Any distance less than Threshold will be zero.
        /// </summary>
        private const float Threshold = 1e-20f;

        private readonly IDeviceChooser _deviceChooser;

        public GpuHalfDistanceDictCalculator(
            IDeviceChooser deviceChooser)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            _deviceChooser = deviceChooser;
        }

        public DodfDistanceContainer CalculateDistances(List<DataItem> fxwList)
        {
            TimeSpan takenTime;

            return
                CreateDistanceDict(fxwList, out takenTime);
        }

        public DodfDistanceContainer CreateDistanceDict(List<DataItem> fxwList, out TimeSpan takenTime)
        {
            var result = new DodfDistanceContainer(fxwList.Count);

            var inputLength = fxwList[0].Input.Length;

            using (var clProvider = new OpenCLDistanceDictHalfProvider(_deviceChooser, true, fxwList, DistanceMemElementCount))
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

inline int ObtainIndex(volatile __global int * indexArray)
{
    return
        atomic_inc(indexArray);
}

{DODF_DISABLE_EXP_DEFINE_CLAUSE}


__kernel void DistanceKernel(
    const __global half * fxwList,
    __global float * distance,
    volatile __global int * indexArray,
    __local float * local_results,

    float threshold,
    int distanceMemLength,
    
    int inputLength,
    int startRowIndex,
    int processedRowCountPerKernelCall,
    int count)
{
    if(get_global_id(0) == 0)
    {
        indexArray[0] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    for (uint cc = startRowIndex + get_group_id(0); cc < startRowIndex + processedRowCountPerKernelCall; cc += get_num_groups(0))
    {
        uint aIndex = cc * inputLength;
        uint aShift = aIndex % 2;

        for (int dd = cc + 1; dd < count; dd++)
        {
            //------------------------ GetExpDistanceDab ------------------------------

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
                
#ifdef DODF_DISABLE_EXP
                float write_result = -result;
                //there is no needs in if clause due to 'debug' mode
#else
                float write_result = exp(-result);
                if(write_result >= threshold)
#endif

                {
                    int index = ObtainIndex(indexArray);
                    distance[index * 3 + 0] = cc;
                    distance[index * 3 + 1] = dd;
                    distance[index * 3 + 2] = write_result;
                }
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

                #region определяем параметры запуска кернела

                //размер локальной группы вычислителей GPU
                const int szLocalSize = 128;

                //количество групп (настроено вручную согласно производительности NVidia GeForce 730M и AMD Radeon 7750
                int szNumGroups = 
                    16 
                    * clProvider.Parameters.NumComputeUnits;

                //глобальный размер
                int szGlobalSize = szNumGroups * szLocalSize;

                #endregion

                var totalItemsCountProcessedByKernel = 0L;
                var totalKernelExcutionCount = 0;

                //Запускаем кернел
                var totalTakenTime = new TimeSpan(0L);

                for (var startRowIndex = 0; startRowIndex < fxwList.Count; )
                {
                    //количество строк, обрабатываемое за один вызов кернела (оно меняется от итерации к итерации из-за того, что крайние (нижние) строки короче
                    int processedRowCountPerKernelCall = DistanceMemElementCount / (fxwList.Count - startRowIndex);

                    if (processedRowCountPerKernelCall < 1)
                    {
                        throw new InvalidOperationException(
                            string.Format(
                                "ProcessedRowCountPerKernelCall is zero due to too big value of fxwList.Count = {0}. Please increase DistanceMemElementCount as low as {0}.",
                                fxwList.Count));
                    }

                    var before = DateTime.Now;

                    #region запуск кернела и ожидание результатов

                    distanceKernel
                        .SetKernelArgMem(0, clProvider.FxwMem)
                        .SetKernelArgMem(1, clProvider.DistanceMem)
                        .SetKernelArgMem(2, clProvider.IndexMem)
                        .SetKernelArgLocalMem(3, 4*szLocalSize)
                        .SetKernelArg(4, 4, Threshold)
                        .SetKernelArg(5, 4, DistanceMemElementCount)
                        .SetKernelArg(6, 4, inputLength)
                        .SetKernelArg(7, 4, startRowIndex)
                        .SetKernelArg(8, 4, processedRowCountPerKernelCall)
                        .SetKernelArg(9, 4, fxwList.Count)
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


                    clProvider.DistanceMem.Read(BlockModeEnum.NonBlocking);
                    clProvider.IndexMem.Read(BlockModeEnum.NonBlocking);

                    // Make sure we're done with everything that's been requested before
                    clProvider.QueueFinish();

                    #endregion

                    var after = DateTime.Now;
                    totalTakenTime += (after - before);

                    totalKernelExcutionCount++;

                    var itemsCountProcessedByKernel = clProvider.IndexMem.Array[0];
                    totalItemsCountProcessedByKernel += itemsCountProcessedByKernel;

                    #region заполняем диктионари

                    for (var cc = 0; cc < itemsCountProcessedByKernel; cc++)
                    {
                        var aIndex = (int)clProvider.DistanceMem.Array[cc * 3 + 0];
                        var bIndex = (int)clProvider.DistanceMem.Array[cc * 3 + 1];
                        var distance = clProvider.DistanceMem.Array[cc * 3 + 2];

                        result.AddValue(aIndex, bIndex, distance);
                    }

                    #endregion
                    
                    //следующая итерация цикла
                    startRowIndex += processedRowCountPerKernelCall;
                }

                var totalItemCount = (long)(fxwList.Count + 1) * fxwList.Count / 2;

                ConsoleAmbientContext.Console.WriteLine(
                    "fxwList count = {0}, total items {1}, processed items {2}, {3}%, others values are below threshold = {4}, with kernel execution count = {5}",
                    fxwList.Count,
                    totalItemCount,
                    totalItemsCountProcessedByKernel,
                    totalItemsCountProcessedByKernel * 10000.0 / totalItemCount / (long)100,
                    Threshold,
                    totalKernelExcutionCount);

                takenTime = totalTakenTime;
            }

            return result;
        }



    }
}
