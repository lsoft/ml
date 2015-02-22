using System;
using MyNN.Common;
using MyNN.Common.Other;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.DeDyAggregator
{
    public class GPUDeDyAggregator : IOpenCLDeDyAggregator
    {
        private const int PreprocessGroupSize = 16;

        private readonly int _previousLayerNeuronCount;
        private readonly int _aggregateLayerNeuronCount;
        private readonly MemFloat _aggregateLayerWeights;

        private readonly int _aggregationFactor;
        private readonly Kernel _preprocessKernel0;
        private readonly Kernel _preprocessKernel1;

        public MemFloat DeDz
        {
            get;
            private set;
        }

        public MemFloat DeDy
        {
            get;
            private set;
        }

        public GPUDeDyAggregator(
            CLProvider clProvider,
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount,
            MemFloat aggregateLayerDeDz,
            MemFloat aggregateLayerWeights
            )
        {
            throw new InvalidOperationException();
        }

        public GPUDeDyAggregator(
            CLProvider clProvider,
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount,
            MemFloat aggregateLayerWeights
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (aggregateLayerWeights == null)
            {
                throw new ArgumentNullException("aggregateLayerWeights");
            }

            if(clProvider.ChoosedDeviceType == DeviceType.Cpu)
            {
                throw new NotSupportedException("Intel CPU is not supported due to bugs in INTEL CPU OPENCL implementation: this aggregation does not work correctly SOMETIMES(!) on Intel CPU, but works fine at ALL THE TIME on INTEL GPU or NVIDIA/AMD GPU. There is some sort of synchronization bug in INTEL CPU OPENCL.");
            }

            _previousLayerNeuronCount = previousLayerNeuronCount;
            _aggregateLayerNeuronCount = aggregateLayerNeuronCount;
            _aggregateLayerWeights = aggregateLayerWeights;

            _aggregationFactor = Helper.UpTo(aggregateLayerNeuronCount, PreprocessGroupSize) / PreprocessGroupSize;

            this.DeDz = clProvider.CreateFloatMem(
                aggregateLayerNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );

            this.DeDy = clProvider.CreateFloatMem(
                previousLayerNeuronCount * _aggregationFactor,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );

            _preprocessKernel0 = clProvider.CreateKernel(
                this.GetAggregationKernel0(PreprocessGroupSize),
                "PreprocessKernel0"
                );

            _preprocessKernel1 = clProvider.CreateKernel(
                this.GetAggregationKernel1(),
                "PreprocessKernel1"
                );


        }

        public void Aggregate(
            )
        {
            _preprocessKernel0
                .SetKernelArgMem(0, this.DeDz)
                .SetKernelArgMem(1, _aggregateLayerWeights)
                .SetKernelArgMem(2, this.DeDy)
                .SetKernelArg(3, sizeof(int), _previousLayerNeuronCount)
                .SetKernelArg(4, sizeof(int), _aggregateLayerNeuronCount)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        Helper.UpTo(_previousLayerNeuronCount, PreprocessGroupSize),
                        Helper.UpTo(_aggregateLayerNeuronCount, PreprocessGroupSize)
                    },
                    new[]
                    {
                        PreprocessGroupSize,
                        PreprocessGroupSize,
                    })
                ;

            var aggregationFactor = _aggregationFactor;

            for (; ; )
            {
                if (aggregationFactor == 1)
                {
                    break;
                }

                int currentLocalGroupSize;
                if (aggregationFactor < PreprocessGroupSize)
                {
                    currentLocalGroupSize = aggregationFactor;
                }
                else
                {
                    currentLocalGroupSize = PreprocessGroupSize;
                }

                var globalx = Helper.UpTo(_previousLayerNeuronCount, currentLocalGroupSize);
                var globaly = Helper.UpTo(aggregationFactor, currentLocalGroupSize);

                _preprocessKernel1
                    .SetKernelArgMem(0, this.DeDy)
                    .SetKernelArg(1, sizeof(int), _previousLayerNeuronCount)
                    .SetKernelArg(2, sizeof(int), aggregationFactor)
                    .SetKernelArg(3, sizeof(int), currentLocalGroupSize)
                    .SetKernelArg(4, sizeof(int), currentLocalGroupSize)
                    .SetKernelArgLocalMem(5, sizeof(float) * currentLocalGroupSize * currentLocalGroupSize)
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

                aggregationFactor = Helper.UpTo(aggregationFactor, currentLocalGroupSize) / currentLocalGroupSize;
            }
        }

        public void ClearAndWrite()
        {
            this.DeDz.Array.Clear();
            this.DeDz.Write(BlockModeEnum.NonBlocking);

            this.DeDy.Array.Clear();
            this.DeDy.Write(BlockModeEnum.NonBlocking);
        }


        #region private code

        private string GetAggregationKernel0(
            int groupSize
            )
        {
            var kernelText = @"
__kernel void PreprocessKernel0(
    __global read_only float * aggregateLayerDeDz,
    __global read_only float * aggregateLayerWeights,
    __global write_only float * gcache,

    int previousNeuronCount,
    int aggregateLayerNeuronCount
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
    if(globalx < previousNeuronCount && globaly < aggregateLayerNeuronCount)
    {
        int aggregateNeuronIndex = globaly;
        int aggregateWeightIndex = aggregateNeuronIndex * previousNeuronCount + globalx;

        float w = aggregateLayerWeights[aggregateWeightIndex];
        float dedz = aggregateLayerDeDz[aggregateNeuronIndex];
        float dedy = w * dedz;

        cache[inCacheIndex] = dedy;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //фаза редукции

    int current_local_size = groupsizey;
    for(int offsety = (groupsizey + 1) / 2; offsety > 0; offsety = (offsety + (offsety > 1 ? 1 : 0)) / 2)
    {
        if (ingry < offsety)
        {
            int other_index = ingry + offsety;
            if(other_index < current_local_size)
            {
                int readIndex = other_index * groupsizex + ingrx;
                int writeIndex = ingry * groupsizex + ingrx;

                cache[writeIndex] += cache[readIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        current_local_size = (current_local_size + 1) / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //если группа не вылазит за пределы MLP
    if(globalx < previousNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * previousNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

";

            kernelText = kernelText.Replace("<GROUP_SIZE>", groupSize.ToString());

            return
                kernelText;
        }

        private string GetAggregationKernel1(
            )
        {
            var kernelText = @"
__kernel void PreprocessKernel1(
    __global float * gcache,

    int previousNeuronCount,
    int aggregateLayerNeuronCount,
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
    if(globalx < previousNeuronCount && globaly < aggregateLayerNeuronCount)
    {
        int aggregateNeuronIndex = globaly;
        int aggregateWeightIndex = aggregateNeuronIndex * previousNeuronCount + globalx;

//        float gvalue = gcache[aggregateWeightIndex];
//        gcache[aggregateWeightIndex] = 0;
//        cache[inCacheIndex] = gvalue;

         // 3 lines up is equivalent with one line below:

        cache[inCacheIndex] = atomic_xchg(gcache + aggregateWeightIndex, 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //фаза редукции

    int current_local_size = groupsizey;
    for(int offsety = (groupsizey + 1) / 2; offsety > 0; offsety = (offsety + (offsety > 1 ? 1 : 0)) / 2)
    {
        if (ingry < offsety)
        {
            int other_index = ingry + offsety;
            if(other_index < current_local_size)
            {
                int readIndex = other_index * groupsizex + ingrx;
                int writeIndex = ingry * groupsizex + ingrx;

                cache[writeIndex] += cache[readIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        current_local_size = (current_local_size + 1) / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //если группа не вылазит за пределы MLP
    if(globalx < previousNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * previousNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

";

            return
                kernelText;
        }


        #endregion
    }
}