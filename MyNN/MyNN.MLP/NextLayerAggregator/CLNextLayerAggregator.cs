using System;
using MyNN.Common;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.NextLayerAggregator
{
    public class CLNextLayerAggregator : INextLayerAggregator
    {
        private const int PreprocessGroupSize = 16;

        private readonly ILayerConfiguration _currentLayer;
        private readonly ILayerConfiguration _nextLayer;
        private readonly MemFloat _nextLayerDeDz;
        private readonly IMemLayerContainer _nextLayerContainer;

        private readonly int _aggregationFactor;
        private readonly Kernel _preprocessKernel0;
        private readonly Kernel _preprocessKernel1;

        public MemFloat PreprocessCache
        {
            get;
            private set;
        }

        public CLNextLayerAggregator(
            CLProvider clProvider,
            ILayerConfiguration currentLayer,
            ILayerConfiguration nextLayer,
            MemFloat nextLayerDeDz,
            IMemLayerContainer nextLayerContainer
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (nextLayer == null)
            {
                throw new ArgumentNullException("nextLayer");
            }
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }
            if (nextLayerContainer == null)
            {
                throw new ArgumentNullException("nextLayerContainer");
            }

            _currentLayer = currentLayer;
            _nextLayer = nextLayer;
            _nextLayerDeDz = nextLayerDeDz;
            _nextLayerContainer = nextLayerContainer;

            _aggregationFactor = Helper.UpTo(nextLayer.TotalNeuronCount, PreprocessGroupSize) / PreprocessGroupSize;

            this.PreprocessCache = clProvider.CreateFloatMem(
                currentLayer.TotalNeuronCount * _aggregationFactor,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );

            _preprocessKernel0 = clProvider.CreateKernel(
                this.GetPreprocessHiddenKernelZeroSource(PreprocessGroupSize),
                "PreprocessKernel0"
                );

            _preprocessKernel1 = clProvider.CreateKernel(
                this.GetPreprocessHiddenKernelOneSource(),
                "PreprocessKernel1"
                );


        }

        public void Aggregate(
            )
        {
            _preprocessKernel0
                .SetKernelArgMem(0, this._nextLayerDeDz)
                .SetKernelArgMem(1, _nextLayerContainer.WeightMem)
                .SetKernelArgMem(2, this.PreprocessCache)
                .SetKernelArg(3, sizeof(int), _currentLayer.TotalNeuronCount)
                .SetKernelArg(4, sizeof(int), _nextLayer.TotalNeuronCount)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        Helper.UpTo(_currentLayer.TotalNeuronCount, PreprocessGroupSize),
                        Helper.UpTo(_nextLayer.TotalNeuronCount, PreprocessGroupSize)
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

                var globalx = Helper.UpTo(_currentLayer.TotalNeuronCount, currentLocalGroupSize);
                var globaly = Helper.UpTo(aggregationFactor, currentLocalGroupSize);

                _preprocessKernel1
                    .SetKernelArgMem(0, this.PreprocessCache)
                    .SetKernelArg(1, sizeof(int), _currentLayer.TotalNeuronCount)
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
            this.PreprocessCache.Array.Clear();
            this.PreprocessCache.Write(BlockModeEnum.NonBlocking);
        }


        #region private code

        private string GetPreprocessHiddenKernelZeroSource(
            int groupSize
            )
        {
            var kernelText = @"
__kernel void PreprocessKernel0(
    __global read_only float * nextLayerDeDz,
    __global read_only float * nextLayerWeights,
    __global write_only float * gcache,

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
    if(globalx < currentNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * currentNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

";

            kernelText = kernelText.Replace("<GROUP_SIZE>", groupSize.ToString());

            return
                kernelText;
        }

        private string GetPreprocessHiddenKernelOneSource(
            )
        {
            var kernelText = @"
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
    if(globalx < currentNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * currentNeuronCount + globalx] = cache[ingrx];
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