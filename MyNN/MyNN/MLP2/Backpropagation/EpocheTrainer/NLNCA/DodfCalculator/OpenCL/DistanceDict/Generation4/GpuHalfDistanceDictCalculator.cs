using System;
using System.Collections.Generic;
using MyNN.Data;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4
{
    /// <summary>
    /// Correct implementation of distance provider for dOdF algorithm that enables GPU-OpenCL with HALF input representation.
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

            using (var clProvider = new Generation4.OpenCLDistanceDictHalfProvider(_deviceChooser, true, fxwList, DistanceMemElementCount))
            {
                var kernelText = @"

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

__kernel void CopyToAccumKernel(
    const __global float * distance,
    __global float * accumulator,
    __local float * local_results,
    long accumShift,
    long totalCount)
{
    //accumulator[get_global_id(0) + accumShift] = distance[get_global_id(0)];

    if(get_global_id(0) < totalCount)
    {
        local_results[get_local_id(0)] = distance[get_global_id(0)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_global_id(0) < totalCount)
    {
        accumulator[get_global_id(0) + accumShift] = local_results[get_local_id(0)];
    }
}
";

#if DODF_DISABLE_EXP
                kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", "#define DODF_DISABLE_EXP");
#else
                kernelText = kernelText.Replace("{DODF_DISABLE_EXP_DEFINE_CLAUSE}", string.Empty);
#endif

                var distanceKernel = clProvider.CreateKernel(kernelText, "DistanceKernel");
                var copyToAccumKernel = clProvider.CreateKernel(kernelText, "CopyToAccumKernel");

                #region определяем параметры запуска кернела

                //размер локальной группы вычислителей GPU
                int szLocalSize;

                //количество групп 
                int szNumGroups;

                //настроено вручную согласно производительности NVidia GeForce 730M и AMD Radeon 7750
                if (clProvider.Parameters.IsVendorAMD)
                {
                    szLocalSize = 64;
                    szNumGroups =
                        256
                        * clProvider.Parameters.NumComputeUnits;

                }
                else
                {
                    //в том числе нвидия
                    szLocalSize = 128;
                    szNumGroups =
                        16
                        * clProvider.Parameters.NumComputeUnits;
                }

                //глобальный размер
                var szGlobalSize = szNumGroups * szLocalSize;

                #endregion

                var totalItemsCountProcessedByKernel = 0L;
                var totalKernelExcutionCount = 0;

                //фактор заполнения массива вычисленными значениями (если вычисленных значений мало, массив clProvider.DistanceMem используется не полностью
                //поэтому можно пропорционально увеличить число обрабатываемых строк за вызов кернела; предполагается, что статистически
                //процент вычисленных значений будет примерно одинаков между строками);
                var fillFactor = 1f;
                var fillFactorCalculated = false;

                //Запускаем кернел
                var totalTakenTime = new TimeSpan(0L);

                for (var startRowIndex = 0; startRowIndex < fxwList.Count; )
                {
                    //количество строк, обрабатываемое за один вызов кернела (оно меняется от итерации к итерации из-за того, что крайние (нижние) строки короче
                    int processedRowCountPerKernelCall = (int) (DistanceMemElementCount/(fxwList.Count - startRowIndex)/fillFactor);

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
                        .SetKernelArgLocalMem(3, sizeof (float)*szLocalSize)
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

                    if (itemsCountProcessedByKernel > DistanceMemElementCount)
                    {
                        throw new Exception("Число обработанных элементов превысило размер массива; результат такой операции не определен. Аварийный останов. Попробуйте отключить fillFactor.");
                    }

                    #region заполняем диктионари

                    for (var cc = 0; cc < itemsCountProcessedByKernel; cc++)
                    {
                        var aIndex = (int) clProvider.DistanceMem.Array[cc*3 + 0];
                        var bIndex = (int) clProvider.DistanceMem.Array[cc*3 + 1];
                        var distance = clProvider.DistanceMem.Array[cc*3 + 2];

                        result.AddValue(aIndex, bIndex, distance);
                    }

                    #endregion

                    #region обновляем фактор заполнения

                    if (!fillFactorCalculated)
                    {
                        if (itemsCountProcessedByKernel > 0)
                        {
                            //20% запас на погрешность
                            fillFactor = Math.Min(1.0f, 1.2f*itemsCountProcessedByKernel/(float) DistanceMemElementCount);

                            fillFactorCalculated = true;
                        }
                    }

                    #endregion

                    #region работаем с аккулумятором

                    if (fillFactorCalculated)
                    {
                        //если аккумулятора нет, то создаем его
                        if (clProvider.AccumMem == null)
                        {
                            var prognosisAboutTotalSize = (long) (fillFactor*GetTotalItemCount(fxwList));

                            clProvider.AllocateAccumulator(prognosisAboutTotalSize);
                        }

                        const int accumCopyLocalSize = 128;
                        long totalCount = itemsCountProcessedByKernel*3;

                        //выполняем кернел копирования
                        copyToAccumKernel
                            .SetKernelArgMem(0, clProvider.DistanceMem)
                            .SetKernelArgMem(1, clProvider.AccumMem)
                            .SetKernelArgLocalMem(2, sizeof(float) * accumCopyLocalSize)
                            .SetKernelArg(3, 8, (long)(totalItemsCountProcessedByKernel * 3))
                            .EnqueueNDRangeKernel(
                                new int[]
                                {
                                    1//totalCount + (accumCopyLocalSize - totalCount % accumCopyLocalSize)
                                },
                                new int[]
                                {
                                    accumCopyLocalSize
                                });

                        clProvider.AccumMem.Read(BlockModeEnum.Blocking);
                    }

                    #endregion

                    //следующая итерация цикла
                    totalItemsCountProcessedByKernel += itemsCountProcessedByKernel;
                    startRowIndex += processedRowCountPerKernelCall;
                }

                var totalItemCount = GetTotalItemCount(fxwList);

                ConsoleAmbientContext.Console.WriteLine(
                    "GPU: fxwList count = {0}, total items {1}, processed items {2}, {3}%, others values are below threshold = {4}, with kernel execution count = {5}",
                    fxwList.Count,
                    totalItemCount,
                    totalItemsCountProcessedByKernel,
                    totalItemsCountProcessedByKernel * 10000.0 / totalItemCount / 100L,
                    Threshold,
                    totalKernelExcutionCount);

                takenTime = totalTakenTime;
            }

            return result;
        }

        private long GetTotalItemCount(List<DataItem> fxwList)
        {
            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            return
                (long)(fxwList.Count + 1) * fxwList.Count / 2;
        }


    }
}
