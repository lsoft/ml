using System;
using System.Collections.Generic;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.OutputConsole;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3.Float
{
    /// <summary>
    /// Distance factory for dOdF algorithm.
    /// This implementation DOES NOT contains optimizations in memory consumption in distance table BUT introduce threshold in distance metric.
    /// It should used for debug-purposes only!
    /// </summary>
    public class VectorizedCpuDistanceDictCalculator : IDistanceDictCalculator
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

        private readonly bool _fillFactorEnable;
        
        public VectorizedCpuDistanceDictCalculator(
            bool fillFactorEnable = true)
        {
            _fillFactorEnable = fillFactorEnable;
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

            using (var clProvider = new OpenCLDistanceDictProvider(new IntelCPUDeviceChooser(), true, fxwList, DistanceMemElementCount))
            {
                var kernelText = @"

inline int ObtainIndex(volatile __global int * indexArray)
{
    return
        atomic_inc(indexArray);
}

{DODF_DISABLE_EXP_DEFINE_CLAUSE}


__kernel void ClearIndexKernel(
        volatile __global int * indexArray)
{
    indexArray[0] = 0;
}

__kernel void DistanceKernel(
    const __global float * fxwList,
    __global float * distance,
    volatile __global int * indexArray,

    float threshold,
    int distanceMemLength,
    
    int inputLength,
    int startRowIndex,
    int processedRowCountPerKernelCall,
    int count)
{
    for (
        uint cc = startRowIndex + get_global_id(0);
        cc < min(count, startRowIndex + processedRowCountPerKernelCall);
        cc += get_global_size(0))
    {
        int ccTail = (cc * inputLength) % 16;
        __global float * fxwCC = fxwList + ccTail;

        for (uint dd = cc + 1; dd < count; dd++)
        {
            //printf(""cc = %i, dd = %i\r\n"", cc, dd);

            //------------------------ GetExpDistanceDab ------------------------------

//////////            float result = 0;
//////////            
//////////            for (int uu = 0; uu < inputLength; uu++)
//////////            {
//////////                float diff = fxwList[cc * inputLength + uu] - fxwList[dd * inputLength + uu];
//////////                result += diff * diff;
//////////            }

            int ddTail = (dd * inputLength) % 16;
            __global float * fxwDD = fxwList + ddTail;

            //нельзя выносить наружу, так как переменные изменяются в цикле ниже
            uint cci = (cc * inputLength) >> 4;
            uint ddi = (dd * inputLength) >> 4;

            //GetExpDistanceDab vectorized
            float16 result16 = 0;

            uint uu = 0;
            for (; uu < inputLength >> 4; uu++, cci++, ddi++)
            {
                float16 fxwA = vload16(
                    cci, 
                    fxwCC);

                float16 fxwB = vload16(
                    ddi, 
                    fxwDD);
    
                float16 diff16 = fxwA - fxwB;
                result16 += diff16 * diff16;
            }
            uu = inputLength - inputLength % 16;

//            uint uu = 0;
//            for (; uu < inputLength - inputLength % 16; uu+=16)
//            {
//                int ccindex = cc * inputLength + uu;
//
//                float16 fxwA = vload16(
//                    ccindex >> 4, 
//                    fxwCC);
//    
//                int ddindex = dd * inputLength + uu;
//                float16 fxwB = vload16(
//                    ddindex >> 4, 
//                    fxwDD);
//    
//                float16 diff16 = fxwA - fxwB;
//                result16 += diff16 * diff16;
//            }
    
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

#ifdef DODF_DISABLE_EXP
            float write_result = -result;
            //there is no needs in if clause due to 'debug' mode
#else
            float write_result = exp(-result);
            if(write_result >= threshold)
#endif

            {
                //printf(""cc = %i, dd = %i\r\n"", cc, dd);

                int index = ObtainIndex(indexArray);
                distance[index * 3 + 0] = cc;
                distance[index * 3 + 1] = dd;
                distance[index * 3 + 2] = write_result;
            }
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
                var clearKernel = clProvider.CreateKernel(kernelText, "ClearIndexKernel");

                #region определяем параметры запуска кернела

                //глобальный размер
                uint szGlobalSize = clProvider.Parameters.NumComputeUnits;

                #endregion

                var totalItemsCountProcessedByKernel = 0L;
                var totalKernelExcutionCount = 0;

                //фактор заполнения массива вычисленными значениями (если вычисленных значений мало, массив clProvider.DistanceMem используется не полностью
                //поэтому можно пропорционально увеличить число обрабатываемых строк за вызов кернела; предполагается, что статистически
                //процент вычисленных значений будет примерно одинаков между строками);
                var fillFactor = 1f;

                //Запускаем кернел
                var totalTakenTime = new TimeSpan(0L);

                for (var startRowIndex = 0; startRowIndex < fxwList.Count; )
                {
                    //количество строк, обрабатываемое за один вызов кернела (оно меняется от итерации к итерации из-за того, что крайние (нижние) строки короче
                    int processedRowCountPerKernelCall = (int)(DistanceMemElementCount / (fxwList.Count - startRowIndex) / fillFactor);

                    if (processedRowCountPerKernelCall < 1)
                    {
                        throw new InvalidOperationException(
                            string.Format(
                                "ProcessedRowCountPerKernelCall is zero due to too big value of fxwList.Count = {0}. Please increase DistanceMemElementCount as low as {0}.",
                                fxwList.Count));
                    }

                    var before = DateTime.Now;

                    #region запуск кернела и ожидание результатов

                    clearKernel
                        .SetKernelArgMem(0, clProvider.IndexMem)
                        .EnqueueNDRangeKernel(1);

                    distanceKernel
                        .SetKernelArgMem(0, clProvider.FxwMem)
                        .SetKernelArgMem(1, clProvider.DistanceMem)
                        .SetKernelArgMem(2, clProvider.IndexMem)
                        .SetKernelArg(3, 4, Threshold)
                        .SetKernelArg(4, 4, DistanceMemElementCount)
                        .SetKernelArg(5, 4, inputLength)
                        .SetKernelArg(6, 4, startRowIndex)
                        .SetKernelArg(7, 4, processedRowCountPerKernelCall)
                        .SetKernelArg(8, 4, fxwList.Count)
                        .EnqueueNDRangeKernel(
                                szGlobalSize
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

                    #region обновляем фактор заполнения

                    if (_fillFactorEnable)
                    {
                        if (startRowIndex == 0)
                        {
                            if (itemsCountProcessedByKernel > 0)
                            {
                                //20% запас на погрешность
                                fillFactor = Math.Min(1.0f, 1.2f * itemsCountProcessedByKernel / (float)DistanceMemElementCount);
                            }
                        }
                    }

                    #endregion

                    //следующая итерация цикла
                    startRowIndex += processedRowCountPerKernelCall;
                }

                var totalItemCount = (long)(fxwList.Count + 1) * fxwList.Count / 2;

                ConsoleAmbientContext.Console.WriteLine(
                    "CPU: fxwList count = {0}, total items {1}, processed items {2}, {3}%, others values are below threshold = {4}, with kernel execution count = {5}",
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
