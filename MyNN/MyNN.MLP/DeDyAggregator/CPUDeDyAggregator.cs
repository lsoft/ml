using System;
using MyNN.Common.Other;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.DeDyAggregator
{
    public class CPUDeDyAggregator : IOpenCLDeDyAggregator
    {
        private readonly int _previousLayerNeuronCount;
        private readonly int _aggregateLayerNeuronCount;
        private readonly MemFloat _aggregateLayerWeights;
        private readonly Kernel _preprocessKernel;

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

        public int TotalNeuronCount
        {
            get
            {
                return
                    _aggregateLayerNeuronCount;
            }
        }

        public CPUDeDyAggregator(
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

            _previousLayerNeuronCount = previousLayerNeuronCount;
            _aggregateLayerNeuronCount = aggregateLayerNeuronCount;
            _aggregateLayerWeights = aggregateLayerWeights;


            this.DeDz = clProvider.CreateFloatMem(
                aggregateLayerNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );

            this.DeDy = clProvider.CreateFloatMem(
                previousLayerNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );

            _preprocessKernel = clProvider.CreateKernel(
                this.GetAggregationKernelSource(),
                "PreprocessKernel0"
                );
        }

        public void Aggregate()
        {
            _preprocessKernel
                .SetKernelArgMem(0, this.DeDz)
                .SetKernelArgMem(1, _aggregateLayerWeights)
                .SetKernelArgMem(2, this.DeDy)
                .SetKernelArg(3, sizeof (int), _previousLayerNeuronCount)
                .SetKernelArg(4, sizeof (int), _aggregateLayerNeuronCount)
                .EnqueueNDRangeKernel(_previousLayerNeuronCount)
                ;
        }

        public void ClearAndWrite()
        {
            this.DeDz.Array.Clear();
            this.DeDz.Write(BlockModeEnum.NonBlocking);

            this.DeDy.Array.Clear();
            this.DeDy.Write(BlockModeEnum.NonBlocking);
        }

        #region private code

        private string GetAggregationKernelSource(
            )
        {
            var r = @"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}


__kernel void PreprocessKernel0(
    __global read_only float * aggregateLayerDeDz,
    __global read_only float * aggregateLayerWeights,
    __global write_only float * gcache, //dedy

    int previousNeuronCount,
    int aggregateLayerNeuronCount
    )
{
    int neuronIndex = get_global_id(0);

    //просчет состо€ни€ нейронов текущего сло€, по состо€нию нейронов последующего (with Kahan Algorithm)
    KahanAccumulator accDeDy = GetEmptyKahanAcc();
    for (int aggregateNeuronIndex = 0; aggregateNeuronIndex < aggregateLayerNeuronCount; ++aggregateNeuronIndex)
    {
        int weightIndex = ComputeWeightIndex(previousNeuronCount, aggregateNeuronIndex) + neuronIndex; //не векторизуетс€:(

        float w = aggregateLayerWeights[weightIndex];
        float dedz = aggregateLayerDeDz[aggregateNeuronIndex];
        float dedy = w * dedz;

        KahanAddElement(&accDeDy, dedy);
    }

    gcache[neuronIndex] = accDeDy.Sum;
}
";

            return r;
        }


        #endregion

    }
}