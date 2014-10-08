using System;
using System.Linq;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4.Sorter
{
    public class AMDBitonicSorter : ISorter
    {
        private readonly Kernel _kernel;

        public AMDBitonicSorter(
            CLProvider clProvider)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _kernel = clProvider.CreateKernel(
                KernelSource,
                "BitonicSortKernel");
        }

        public void Sort(
            MemByte dataMem,
            ulong totalElementCountPlusOverhead
            )
        {
            if (dataMem == null)
            {
                throw new ArgumentNullException("dataMem");
            }

            if (totalElementCountPlusOverhead < 4)
            {
                throw new ArgumentOutOfRangeException("totalElementCountPlusOverhead");
            }

            var log2d = Math.Log(totalElementCountPlusOverhead, 2);
            if ((log2d % 1) > double.Epsilon)
            {
                throw new ArgumentException(
                    "Element amount should be equal of any power of 2.",
                    "totalElementCountPlusOverhead");
            }

            var localSizes = new uint[]
            {
                2,
                4,
                8,
                16,
                32,
                64,
                128
            };

            var localSize = localSizes
                .Where(j => (2 * j) <= totalElementCountPlusOverhead)
                .Max();

            //выполняем алгоритм

            uint numStages = 0;
            for (var temp = totalElementCountPlusOverhead; temp > 1; temp >>= 1)
            {
                ++numStages;
            }

            var globalThreads = new ulong[] { totalElementCountPlusOverhead / 2 };
            var localThreads = new ulong[] { localSize };

            for (var stage = 0; stage < numStages; ++stage)
            {
                // Every stage has stage + 1 passes
                for (var passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
                {
                    //*
                    //* Enqueue a kernel run call.
                    //* For simplicity, the groupsize used is 1.
                    //*
                    //* Each thread writes a sorted pair.
                    //* So, the number of  threads (global) is half the length.
                    //*
                    _kernel
                        .SetKernelArgMem(0, dataMem)
                        .SetKernelArg(1, sizeof(uint), stage)
                        .SetKernelArg(2, sizeof(uint), passOfStage)
                        .SetKernelArg(3, sizeof(uint), (uint)1)
                        .EnqueueNDRangeKernel(
                            globalThreads,
                            localThreads);
                }
            }
        }

        private const string KernelSource = @"
typedef struct
{
    uint AIndex;
    uint BIndex;
    float Distance;
} SortItem;

__inline ulong GetKey(SortItem d)
{
    ulong a = d.AIndex;

    return
        (a << 32) + d.BIndex;
}

__kernel void BitonicSortKernel(
    __global SortItem * theArray,
    const uint stage, 
    const uint passOfStage,
    const uint direction
    )
{
    uint sortIncreasing = direction;
    ulong threadId = get_global_id(0);
     
    ulong pairDistance = 1 << (stage - passOfStage);
    ulong blockWidth   = 2 * pairDistance;
 
    ulong leftId = (threadId % pairDistance) 
                   + (threadId / pairDistance) * blockWidth;
 
    ulong rightId = leftId + pairDistance;
     
    SortItem leftElement = theArray[leftId];
    SortItem rightElement = theArray[rightId];
     
    ulong sameDirectionBlockWidth = 1 << stage;
     
    if((threadId/sameDirectionBlockWidth) % 2 == 1)
    {
        sortIncreasing = 1 - sortIncreasing;
    }
 
    SortItem greater;
    SortItem lesser;
    if(GetKey(leftElement) > GetKey(rightElement))
    {
        greater = leftElement;
        lesser  = rightElement;
    }
    else
    {
        greater = rightElement;
        lesser  = leftElement;
    }
     
    if(sortIncreasing)
    {
        theArray[leftId]  = lesser;
        theArray[rightId] = greater;
    }
    else
    {
        theArray[leftId]  = greater;
        theArray[rightId] = lesser;
    }
}
";

    }
}