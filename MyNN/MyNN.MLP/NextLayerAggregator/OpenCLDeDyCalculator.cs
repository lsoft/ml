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
    public class OpenCLDeDyCalculator : IOpenCLDeDyCalculator
    {
        private const int PreprocessGroupSize = 16;

        private readonly int _currentLayerTotalNeuronCount;
        private readonly int _nextLayerTotalNeuronCount;
        private readonly MemFloat _nextLayerDeDz;
        private readonly MemFloat _nextLayerWeights;

        private readonly int _aggregationFactor;
        private readonly Kernel _preprocessKernel0;
        private readonly Kernel _preprocessKernel1;

        public MemFloat DeDy
        {
            get;
            private set;
        }

        public OpenCLDeDyCalculator(
            CLProvider clProvider,
            int currentLayerTotalNeuronCount,
            int nextLayerTotalNeuronCount,
            MemFloat nextLayerDeDz,
            MemFloat nextLayerWeights
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }
            if (nextLayerWeights == null)
            {
                throw new ArgumentNullException("nextLayerWeights");
            }

            if(clProvider.ChoosedDeviceType == DeviceType.Cpu)
            {
                throw new NotSupportedException("Intel CPU is not supported due to bugs in INTEL CPU OPENCL implementation: this aggregation does not work correctly SOMETIMES(!) on Intel CPU, but works fine at ALL THE TIME on INTEL GPU or NVIDIA/AMD GPU. There is some sort of synchronization bug in INTEL CPU OPENCL.");
            }

            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;
            _nextLayerTotalNeuronCount = nextLayerTotalNeuronCount;
            _nextLayerDeDz = nextLayerDeDz;
            _nextLayerWeights = nextLayerWeights;

            _aggregationFactor = Helper.UpTo(nextLayerTotalNeuronCount, PreprocessGroupSize) / PreprocessGroupSize;

            this.DeDy = clProvider.CreateFloatMem(
                currentLayerTotalNeuronCount * _aggregationFactor,
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
                .SetKernelArgMem(1, _nextLayerWeights)
                .SetKernelArgMem(2, this.DeDy)
                .SetKernelArg(3, sizeof(int), _currentLayerTotalNeuronCount)
                .SetKernelArg(4, sizeof(int), _nextLayerTotalNeuronCount)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        Helper.UpTo(_currentLayerTotalNeuronCount, PreprocessGroupSize),
                        Helper.UpTo(_nextLayerTotalNeuronCount, PreprocessGroupSize)
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

                var globalx = Helper.UpTo(_currentLayerTotalNeuronCount, currentLocalGroupSize);
                var globaly = Helper.UpTo(aggregationFactor, currentLocalGroupSize);

                _preprocessKernel1
                    .SetKernelArgMem(0, this.DeDy)
                    .SetKernelArg(1, sizeof(int), _currentLayerTotalNeuronCount)
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
            this.DeDy.Array.Clear();
            this.DeDy.Write(BlockModeEnum.NonBlocking);
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

    //���� ������ �� ������� �� ������� MLP
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

    //���� ��������

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

    //���� ������ �� ������� �� ������� MLP
    if(globalx < currentNeuronCount)
    {
        //����� � ���������� ���
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

    //���� ������ �� ������� �� ������� MLP
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

    //���� ��������

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

    //���� ������ �� ������� �� ������� MLP
    if(globalx < currentNeuronCount)
    {
        //����� � ���������� ���
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