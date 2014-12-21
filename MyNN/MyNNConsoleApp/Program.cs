using System;
using System.CodeDom;
using MathNet.Numerics.Integration.Algorithms;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNNConsoleApp.DBN;
using MyNNConsoleApp.RefactoredForDI;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using OpenCvSharp.CPlusPlus.Flann;

namespace MyNNConsoleApp
{
    class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                //A.Do();
                CompareBPGPU.DoCompare();


                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }


    class A
    {
        public static void Do(
            )
        {
            //const int currentLayerNeuronCount = 2;
            //const int nextLayerNeuronCount = 5;
            const int currentLayerNeuronCount = 1001;
            const int nextLayerNeuronCount = 1000;

            Func<CLProvider, MemFloat> getDeDz = (clProvider) =>
            {
                var random = new DefaultRandomizer(123);

                var nextLayerDeDz = clProvider.CreateFloatMem(
                    nextLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );
                //nextLayerDeDz.Array.Fill((int index) => 1000f);
                //nextLayerDeDz.Array.Fill(j => ((int) (random.Next()*10)));
                //nextLayerDeDz.Array.Fill(j => ((int)(random.Next() * 100000)) / 100f);
                nextLayerDeDz.Array.Fill(j => random.Next());
                nextLayerDeDz.Write(BlockModeEnum.Blocking);

                return
                    nextLayerDeDz;
            };

            Func<CLProvider, MemFloat> getWeights = (clProvider) =>
            {
                var random = new DefaultRandomizer(123);

                var nextLayerWeights = clProvider.CreateFloatMem(
                    currentLayerNeuronCount * nextLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );
                //nextLayerWeights.Array.Fill((int index) => (float)index);
                nextLayerWeights.Array.Fill(j => random.Next());
                //nextLayerWeights.Array.Fill(j => ((int) (random.Next()*10)));
                //nextLayerWeights.Array.Fill(j => ((int)(random.Next() * 100000)) / 100f);
                nextLayerWeights.Write(BlockModeEnum.Blocking);

                //foreach (var v in nextLayerWeights.Array)
                //{
                //    Console.WriteLine(DoubleConverter.ToExactString(v));
                //}

                return
                    nextLayerWeights;
            };

            Func<CLProvider, MemFloat> getResults = (clProvider) =>
            {
                var results = clProvider.CreateFloatMem(
                    currentLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                return results;
            };

            for (var cycle = 0; cycle < 100; cycle++)
            {
                var origd = DoOrig(
                    getDeDz,
                    getWeights,
                    getResults,
                    currentLayerNeuronCount,
                    nextLayerNeuronCount
                    );

                var newd = DoNew(
                    getDeDz,
                    getWeights,
                    getResults,
                    currentLayerNeuronCount,
                    nextLayerNeuronCount
                    );

                //var firstNeuronSum = 0f;
                //for (var cc = 0; cc < nextLayerNeuronCount; cc++)
                //{
                //    firstNeuronSum += cc*currentLayerNeuronCountNonBias;
                //}


                for (var cc = 0; cc < origd.Length; cc++)
                {
                    Console.WriteLine(
                        "ORID NEWD: {0}                {1}",
                        DoubleConverter.ToExactString(origd[cc]),
                        DoubleConverter.ToExactString(newd[cc])
                        );
                }

                float maxDiff;
                int maxDiffIndex;
                if (ArrayOperations.ValuesAreEqual(origd, newd, 1e-3f, out maxDiff, out maxDiffIndex))
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("OK, MAX DIFF {0}", maxDiff);
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("MAX DIFF {0}", maxDiff);
                }

                Console.ResetColor();
            }
        }

        private static float[] DoNew(
            Func<CLProvider, MemFloat> getDeDz,
            Func<CLProvider, MemFloat> getWeights,
            Func<CLProvider, MemFloat> getResults,
            int currentLayerNeuronCountNonBias,
            int nextLayerNeuronCount
            )
        {
            using (var clProvider = new CLProvider(
                //new IntelCPUDeviceChooser(false), true))
                new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var nextLayerDeDz = getDeDz(clProvider);
                var nextLayerWeights = getWeights(clProvider);
                var results = getResults(clProvider);

                var kernelText0 = @"
__kernel void PreprocessKernel0(
    __global float * nextLayerDeDz,
    __global float * nextLayerWeights,
    __global float * gcache,

    int currentNeuronCount,
    int nextLayerNeuronCount
    )
{
    const int groupsizex = <GROUP_SIZE>;
    const int groupsizey = <GROUP_SIZE>;

    __local float cache[<GROUP_SIZE> * <GROUP_SIZE>];

    int globalx = get_global_id(0);
    int globaly = get_global_id(1);

    int groupx = get_group_id(0);
    int groupy = get_group_id(1);

    int ingrx = get_local_id(0);
    int ingry = get_local_id(1);

    int inCacheIndex = ingry * groupsizex + ingrx;
    cache[inCacheIndex] = 0;

    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount && globaly < nextLayerNeuronCount)
    {
        int nextNeuronIndex = globaly;
        int nextWeightIndex = nextNeuronIndex * currentNeuronCount + globalx;
        
        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        cache[inCacheIndex] = multiplied;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //фаза редукции

//    int current_local_size = groupsizey;
//    for(int offsety = (groupsizey + 1) / 2; offsety > 0; offsety = (offsety + (offsety > 1 ? 1 : 0)) / 2)
//    {
//        if (ingry < offsety)
//        {
//            int other_index = ingry + offsety;
//            if(other_index < current_local_size)
//            {
//                int readIndex = other_index * groupsizex + ingrx;
//                int writeIndex = ingry * groupsizex + ingrx;
//
//                cache[writeIndex] += cache[readIndex];
//            }
//        }
//
//        barrier(CLK_LOCAL_MEM_FENCE);
//
//        current_local_size = (current_local_size + 1) / 2;
//    }
//
//    barrier(CLK_LOCAL_MEM_FENCE);




    if(ingrx == 0 && ingry == 0)
    {
        for(int ix = 0; ix < groupsizex; ix++)
        {
            KahanAccumulator acc = GetEmptyKahanAcc();
            //float acc = 0;
            
            for(int iy = 0; iy < groupsizey; iy++)
            {
                //cache[ix] += cache[ix + groupsizey * iy];
                KahanAddElement(&acc, cache[ix + groupsizey * iy]);
                
            }

            cache[ix] = acc.Sum;
            //cache[ix] = acc;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);




    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * currentNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

";

                var kernelText1 = @"
__kernel void PreprocessKernel1(
    __global float * gcache,

    int currentNeuronCount,
    int nextLayerNeuronCount,
    int groupsizex,
    int groupsizey,

    __local float * cache
    )
{
    int globalx = get_global_id(0);
    int globaly = get_global_id(1);

    int groupx = get_group_id(0);
    int groupy = get_group_id(1);

    int ingrx = get_local_id(0);
    int ingry = get_local_id(1);

    int inCacheIndex = ingry * groupsizex + ingrx;
    cache[inCacheIndex] = 0;

    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount && globaly < nextLayerNeuronCount)
    {
        int nextNeuronIndex = globaly;
        int nextWeightIndex = nextNeuronIndex * currentNeuronCount + globalx;

//        float gvalue = gcache[nextWeightIndex];
//        gcache[nextWeightIndex] = 0;
//        cache[inCacheIndex] = gvalue;

         // 3 lines up is equivalent with one line below:

        cache[inCacheIndex] = atomic_xchg(gcache + nextWeightIndex, 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //фаза редукции

//    int current_local_size = groupsizey;
//    for(int offsety = (groupsizey + 1) / 2; offsety > 0; offsety = (offsety + (offsety > 1 ? 1 : 0)) / 2)
//    {
//        if (ingry < offsety)
//        {
//            int other_index = ingry + offsety;
//            if(other_index < current_local_size)
//            {
//                int readIndex = other_index * groupsizex + ingrx;
//                int writeIndex = ingry * groupsizex + ingrx;
//
//                cache[writeIndex] += cache[readIndex];
//            }
//        }
//
//        barrier(CLK_LOCAL_MEM_FENCE);
//
//        current_local_size = (current_local_size + 1) / 2;
//    }
//
//    barrier(CLK_LOCAL_MEM_FENCE);



    if(ingrx == 0 && ingry == 0)
    {
        for(int ix = 0; ix < groupsizex; ix++)
        {
            KahanAccumulator acc = GetEmptyKahanAcc();
            
            for(int iy = 0; iy < groupsizey; iy++)
            {
                //cache[ix] += cache[ix + groupsizey * iy];
                KahanAddElement(&acc, cache[ix + groupsizey * iy]);
            }

            cache[ix] = acc.Sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);




    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * currentNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
";

                const int groupSize = 16;

                var aggregationFactor = UpTo(nextLayerNeuronCount, groupSize)/groupSize;

                var gcache = clProvider.CreateFloatMem(
                    currentLayerNeuronCountNonBias*aggregationFactor,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                gcache.Array.Clear();
                gcache.Write(BlockModeEnum.Blocking);

                kernelText0 = kernelText0.Replace("<GROUP_SIZE>", groupSize.ToString());

                var kernel0 = clProvider.CreateKernel(
                    kernelText0,
                    "PreprocessKernel0");

                var kernel1 = clProvider.CreateKernel(
                    kernelText1,
                    "PreprocessKernel1");


                clProvider.QueueFinish();

                var before = DateTime.Now;

                kernel0
                    .SetKernelArgMem(0, nextLayerDeDz)
                    .SetKernelArgMem(1, nextLayerWeights)
                    .SetKernelArgMem(2, gcache)
                    .SetKernelArg(3, sizeof (int), currentLayerNeuronCountNonBias)
                    .SetKernelArg(4, sizeof (int), nextLayerNeuronCount)
                    .EnqueueNDRangeKernel(
                        new[]
                        {
                            UpTo(currentLayerNeuronCountNonBias, groupSize),
                            UpTo(nextLayerNeuronCount, groupSize)
                        },
                        new[]
                        {
                            groupSize,
                            groupSize,
                        })
                    ;

                ////////////////////////////////////////////////////////////////////////////////////////////////////////


                gcache.Read(BlockModeEnum.Blocking);


                //foreach (var v in gcache.Array)
                //{
                //    Console.WriteLine(DoubleConverter.ToExactString(v));
                //}


                for (var cc = 0; cc < results.Array.Length; cc++)
                {
                    var acc = new KahanAlgorithm.Accumulator();
                    //var acc = 0f;
                    //var acc = 0.0;

                    for (var dd = cc; dd < gcache.Array.Length; dd += currentLayerNeuronCountNonBias)
                    {
                        KahanAlgorithm.AddElement(ref acc, gcache.Array[dd]);
                        //acc += gcache.Array[dd];
                        //acc += (double)gcache.Array[dd];
                    }

                    results.Array[cc] = acc.Sum;
                    //results.Array[cc] = acc;
                    //results.Array[cc] = (float)acc;
                }

                return results.Array.CloneArray();

                ////////////////////////////////////////////////////////////////////////////////////////////////////////

                for (;;)
                {
                    //gcache.Read(BlockModeEnum.Blocking);

                    if (aggregationFactor == 1)
                    {
                        break;
                    }

                    int currentLocalGroupSize;
                    if (aggregationFactor < groupSize)
                    {
                        currentLocalGroupSize = aggregationFactor;
                    }
                    else
                    {
                        currentLocalGroupSize = groupSize;
                    }


                    var globalx = UpTo(currentLayerNeuronCountNonBias, currentLocalGroupSize);
                    var globaly = UpTo(aggregationFactor, currentLocalGroupSize);


                    kernel1
                        .SetKernelArgMem(0, gcache)
                        .SetKernelArg(1, sizeof (int), currentLayerNeuronCountNonBias)
                        .SetKernelArg(2, sizeof (int), aggregationFactor)
                        .SetKernelArg(3, sizeof (int), currentLocalGroupSize)
                        .SetKernelArg(4, sizeof (int), currentLocalGroupSize)
                        .SetKernelArgLocalMem(5, sizeof (float)*currentLocalGroupSize*currentLocalGroupSize)
                        .EnqueueNDRangeKernel(
                            new[]
                            {
                                globalx,
                                globaly
                            },
                            new[]
                            {
                                currentLocalGroupSize,
                                currentLocalGroupSize,
                            })
                        ;

                    aggregationFactor = UpTo(aggregationFactor, currentLocalGroupSize)/currentLocalGroupSize;
                }

                clProvider.QueueFinish();

                var after = DateTime.Now;

                Console.WriteLine("NEW METHODS TAKES {0}", (after - before));

                gcache.Read(BlockModeEnum.Blocking);

                Array.Copy(
                    gcache.Array,
                    0,
                    results.Array,
                    0,
                    results.Array.Length
                    );

                return results.Array.CloneArray();
            }
        }

        private static float[] DoOrig(
            Func<CLProvider, MemFloat> getDeDz,
            Func<CLProvider, MemFloat> getWeights,
            Func<CLProvider, MemFloat> getResults,
            int currentLayerNeuronCountNonBias,
            int nextLayerNeuronCount
            )
        {
            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var nextLayerDeDz = getDeDz(clProvider);
                var nextLayerWeights = getWeights(clProvider);
                var results = getResults(clProvider);

                for (int neuronIndex = 0; neuronIndex < currentLayerNeuronCountNonBias; neuronIndex++)
                {
                    // просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)

                    //var accDeDz = new KahanAlgorithm.Accumulator();
                    //float acc = 0;
                    double acc = 0;
                    for (
                        int nextNeuronIndex = 0;
                        nextNeuronIndex < nextLayerNeuronCount;
                        nextNeuronIndex += 1
                        )
                    {
                        int nextWeightIndex =
                            ComputeWeightIndex(currentLayerNeuronCountNonBias, nextNeuronIndex) +
                            neuronIndex;

                        var nextWeight = (double)nextLayerWeights.Array[nextWeightIndex];
                        var nextNabla = (double)nextLayerDeDz.Array[nextNeuronIndex];
                        var multiplied = nextWeight * nextNabla;

                        //KahanAlgorithm.AddElement(ref accDeDz, multiplied);
                        //acc += multiplied;
                        acc += multiplied;
                    }

                    //results.Array[neuronIndex] = accDeDz.Sum;
                    //results.Array[neuronIndex] = acc;
                    results.Array[neuronIndex] = (float)acc;
                }

                return
                    results.Array.CloneArray();
            }
        }

        static int ComputeWeightIndex(
            int previousLayerNeuronCount,
            int neuronIndex)
        {
            return
                previousLayerNeuronCount * neuronIndex;
        }


/*
        private static float[] DoOrig(
            Func<CLProvider, MemFloat> getDeDz,
            Func<CLProvider, MemFloat> getWeights,
            Func<CLProvider, MemFloat> getResults,
            int currentLayerNeuronCountNonBias,
            int nextLayerNeuronCount
            )
        {
            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var nextLayerDeDz = getDeDz(clProvider);
                var nextLayerWeights = getWeights(clProvider);
                var results = getResults(clProvider);

                var origKernel = clProvider.CreateKernel(@"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

__kernel void HiddenLayerTrain(
    __global float * nextLayerDeDz,
    __global float * nextLayerWeights,
    __global float * results,

    int currentLayerNeuronCountNonBias,
    int nextLayerNeuronCount

    )
{
    int neuronIndex = get_global_id(0);

    // просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)

    //KahanAccumulator accDeDz = GetEmptyKahanAcc();
    float acc = 0;
    for (
        int nextNeuronIndex = 0;
        nextNeuronIndex < nextLayerNeuronCount; 
        nextNeuronIndex += 1
        )
    {
        int nextWeightIndex = 
            ComputeWeightIndex(currentLayerNeuronCountNonBias + 1, nextNeuronIndex) + 
            neuronIndex;

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        //KahanAddElement(&accDeDz, multiplied);
        acc += multiplied;
    }

    //results[neuronIndex] = accDeDz.Sum;
    results[neuronIndex] = acc;
}

",
                    "HiddenLayerTrain");

                clProvider.QueueFinish();

                var before = DateTime.Now;

                origKernel
                    .SetKernelArgMem(0, nextLayerDeDz)
                    .SetKernelArgMem(1, nextLayerWeights)
                    .SetKernelArgMem(2, results)
                    .SetKernelArg(3, sizeof(int), currentLayerNeuronCountNonBias - 1)
                    .SetKernelArg(4, sizeof(int), nextLayerNeuronCount)
                    .EnqueueNDRangeKernel(
                        currentLayerNeuronCountNonBias
                        )
                    ;

                clProvider.QueueFinish();

                var after = DateTime.Now;

                Console.WriteLine("OLD METHODS TAKES {0}", (after - before));

                results.Read(BlockModeEnum.Blocking);

                return results.Array.CloneArray();
            }
        }
        //*/

/*
        private static float[] DoOrig(
            Func<CLProvider, MemFloat> getDeDz,
            Func<CLProvider, MemFloat> getWeights,
            Func<CLProvider, MemFloat> getResults,
            int currentLayerNeuronCountNonBias, 
            int nextLayerNeuronCount
            )
        {
            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var nextLayerDeDz = getDeDz(clProvider);
                var nextLayerWeights = getWeights(clProvider);
                var results = getResults(clProvider);

                var origKernel = clProvider.CreateKernel(@"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

__kernel void HiddenLayerTrain(
    __global float * nextLayerDeDz,
    __global float * nextLayerWeights,
    __global float * results,

    int currentLayerNeuronCountNonBias,
    int nextLayerNeuronCount,

    __local float * local_accum

    )
{
    int neuronIndex = get_group_id(0);

    // просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)

    //KahanAccumulator accDeDz = GetEmptyKahanAcc();
    float acc = 0;
    for (
        int nextNeuronIndex = get_local_id(0);
        nextNeuronIndex < nextLayerNeuronCount; 
        nextNeuronIndex += get_local_size(0)
        )
    {
        int nextWeightIndex = 
            ComputeWeightIndex(currentLayerNeuronCountNonBias + 1, nextNeuronIndex) + 
            neuronIndex;

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        //KahanAddElement(&accDeDz, multiplied);
        acc += multiplied;
    }

    //local_accum[get_local_id(0)] = accDeDz.Sum;
    local_accum[get_local_id(0)] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(local_accum);
    barrier(CLK_LOCAL_MEM_FENCE);
    float currentDeDz = local_accum[0];

    if(get_local_id(0) == 0)
    {
        results[neuronIndex] = currentDeDz;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

",
                    "HiddenLayerTrain");

                const uint HiddenLocalGroupSize = 16;
                uint HiddenGlobalGroupSize =
                    (uint) currentLayerNeuronCountNonBias*HiddenLocalGroupSize
                    ;

                clProvider.QueueFinish();

                var before = DateTime.Now;

                origKernel
                    .SetKernelArgMem(0, nextLayerDeDz)
                    .SetKernelArgMem(1, nextLayerWeights)
                    .SetKernelArgMem(2, results)
                    .SetKernelArg(3, sizeof (int), currentLayerNeuronCountNonBias - 1)
                    .SetKernelArg(4, sizeof (int), nextLayerNeuronCount)
                    .SetKernelArgLocalMem(5, 4*HiddenLocalGroupSize)
                    .EnqueueNDRangeKernel(
                        new[]
                        {
                            HiddenGlobalGroupSize
                        },
                        new[]
                        {
                            HiddenLocalGroupSize
                        })
                    ;

                clProvider.QueueFinish();

                var after = DateTime.Now;

                Console.WriteLine("OLD METHODS TAKES {0}", (after - before));

                results.Read(BlockModeEnum.Blocking);

                return results.Array.CloneArray();
            }
        }
//*/

                    private static
                    int UpTo(int value, int step)
        {
            if (value < step)
            {
                return step;
            }

            var ostatok = value%step;

            if (ostatok == 0)
            {
                return value;
            }

            return
                value + step - ostatok;

        }

    }

}
