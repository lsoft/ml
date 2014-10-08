using System;
using System.Collections.Generic;
using MyNN.Common.Data;
using MyNN.Common.OutputConsole;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4.Sorter;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4
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
        /// Distance Mem object length in element (in bytes it will be bigger 12 times (see StructureSizeOf)
        /// </summary>
        private const uint DistanceMemElementCount = 1024 * 1024 * 10;

        /// <summary>
        /// Threshold for distance.
        /// Any distance less than Threshold will be zero.
        /// </summary>
        private const float Threshold = 1e-20f;

        /// <summary>
        /// Sizeof of calculated structure
        /// </summary>
        private const uint StructureSizeOf = (sizeof(uint) * 2 + sizeof(float));

        private readonly IDeviceChooser _deviceChooser;
        private readonly ISorterFactory _sorterFactory;

        public GpuHalfDistanceDictCalculator(
            IDeviceChooser deviceChooser,
            ISorterFactory sorterFactory)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }
            if (sorterFactory == null)
            {
                throw new ArgumentNullException("sorterFactory");
            }

            _deviceChooser = deviceChooser;
            _sorterFactory = sorterFactory;
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

inline uint ObtainIndex(volatile __global uint * indexArray)
{
    return
        atomic_inc(indexArray);
}

{DODF_DISABLE_EXP_DEFINE_CLAUSE}


typedef struct
{
    uint AIndex;
    uint BIndex;
    float Distance;
} Item;


__kernel void DistanceKernel(
    const __global half * fxwList,
    __global Item * distance,
    volatile __global uint * indexArray,
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
                    distance[index].AIndex = cc;
                    distance[index].BIndex = dd;
                    distance[index].Distance = write_result;
                }
            }

            // Synchronize to make sure the first work-item is done with
            // reading partialDotProduct
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

__kernel void CopyToAccumKernel(
    const __global Item * distance,
    __global Item * accumulator,
    ulong accumShift,
    ulong totalCount)
{
    if(get_global_id(0) < totalCount)
    {
        accumulator[get_global_id(0) + accumShift] = distance[get_global_id(0)];
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
                uint szLocalSize;

                //количество групп 
                uint szNumGroups;

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

                ulong totalItemsCountProcessedByKernel = 0L;
                uint totalKernelExcutionCount = 0;

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
                    processedRowCountPerKernelCall = Math.Min(processedRowCountPerKernelCall, fxwList.Count - startRowIndex);
                    ulong processedElementCount = GetTotalItemCount(fxwList.Count - startRowIndex) - GetTotalItemCount(fxwList.Count - (startRowIndex + processedRowCountPerKernelCall));

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
                            new []
                            {
                                szGlobalSize
                            }
                            , new []
                            {
                                szLocalSize
                            }
                        );


                    clProvider.DistanceMem.Read(BlockModeEnum.NonBlocking);
                    clProvider.IndexMem.Read(BlockModeEnum.NonBlocking);

                    // Make sure we're done with everything that's been requested before
                    clProvider.QueueFinish();

                    #endregion

                    totalKernelExcutionCount++;

                    var itemsCountProcessedByKernel = clProvider.IndexMem.Array[0];

                    if (itemsCountProcessedByKernel > DistanceMemElementCount)
                    {
                        throw new Exception("Число обработанных элементов превысило размер массива; результат такой операции не определен. Аварийный останов. Попробуйте отключить fillFactor.");
                    }

                    #region обновляем фактор заполнения

                    if (!fillFactorCalculated)
                    {
                        if (itemsCountProcessedByKernel > 0)
                        {
                            //20% запас на погрешность
                            fillFactor = (float)Math.Min(1.0, 1.2 * itemsCountProcessedByKernel / (double)processedElementCount);

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
                            var prognosisAboutTotalSize = (ulong) (fillFactor*GetTotalItemCount(fxwList.Count));

                            var log2d = Math.Log(prognosisAboutTotalSize, 2);
                            var log2 = ((log2d%1) > double.Epsilon) ? (ulong) (log2d + 1) : (ulong) log2d;

                            var powerOf2AbovePrognosis = (ulong) Math.Pow(2, log2);

                            clProvider.AllocateAccumulator(powerOf2AbovePrognosis);

                            clProvider.AccumMem.Write(BlockModeEnum.Blocking);
                        }

                        const uint accumCopyLocalSize = 128;
                        ulong totalCount = itemsCountProcessedByKernel;

                        //выполняем кернел копирования
                        copyToAccumKernel
                            .SetKernelArgMem(0, clProvider.DistanceMem)
                            .SetKernelArgMem(1, clProvider.AccumMem)
                            .SetKernelArg(2, sizeof(ulong), totalItemsCountProcessedByKernel)
                            .SetKernelArg(3, sizeof(ulong), totalCount)
                            .EnqueueNDRangeKernel(
                                new []
                                {
                                    (ulong)(totalCount + (accumCopyLocalSize - totalCount % accumCopyLocalSize))
                                }
                                ,new []
                                {
                                    (ulong)accumCopyLocalSize
                                }
                                );
                    }

                    #endregion

                    var after = DateTime.Now;
                    totalTakenTime += (after - before);

                    //следующая итерация цикла
                    totalItemsCountProcessedByKernel += itemsCountProcessedByKernel;
                    startRowIndex += processedRowCountPerKernelCall;
                }

                #region сортировка

                var sorter = _sorterFactory.CreateSorter(clProvider);

                sorter.Sort(
                    clProvider.AccumMem,
                    clProvider.AccumulatorActualItemCount
                    );

                #endregion

                #region читаем результат и заполняем диктионари

                if (clProvider.AccumMem != null)
                {
                    clProvider.AccumMem.Read(BlockModeEnum.Blocking);

                    var tmpArray = new byte[StructureSizeOf];
                    for (ulong cc = 0; cc < totalItemsCountProcessedByKernel; cc++)
                    {
                        Array.Copy(clProvider.AccumMem.Array, (long)cc * StructureSizeOf, tmpArray, 0L, StructureSizeOf);

                        var aIndex = BitConverter.ToUInt32(tmpArray, 0);
                        var bIndex = BitConverter.ToUInt32(tmpArray, 4);
                        var distance = BitConverter.ToSingle(tmpArray, 8);

                        result.AddValue(
                            (int)aIndex, 
                            (int)bIndex,
                            distance);
                    }

                    //for (var cc = 0; cc < totalItemsCountProcessedByKernel; cc++)
                    //{
                    //    var aIndex = BitConverter.ToInt32(clProvider.AccumMem.Array, cc*(sizeof(int) * 2 + sizeof(float)) + 0);
                    //    var bIndex = BitConverter.ToInt32(clProvider.AccumMem.Array, cc * (sizeof(int) * 2 + sizeof(float)) + sizeof(int));
                    //    var distance = BitConverter.ToSingle(clProvider.AccumMem.Array, cc * (sizeof(int) * 2 + sizeof(float)) + 2 * sizeof(int));

                    //    result.AddValue(aIndex, bIndex, distance);
                    //}
                }

                #endregion


                var totalItemCount = GetTotalItemCount(fxwList.Count);

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

        private ulong GetTotalItemCount(int count)
        {
            //count == 0 is allowed value

            if (count < 0)
            {
                throw new ArgumentOutOfRangeException("count");
            }
            
            var ucount = (uint)count;

            return
                (ulong)(ucount + 1) * ucount / 2;
        }


    }
}
