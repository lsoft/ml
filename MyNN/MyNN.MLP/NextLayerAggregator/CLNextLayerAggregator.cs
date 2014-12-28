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
            IKernelTextProvider kernelTextProvider,
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
            if (kernelTextProvider == null)
            {
                throw new ArgumentNullException("kernelTextProvider");
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

            _aggregationFactor = Helper.UpTo(nextLayer.NonBiasNeuronCount, PreprocessGroupSize) / PreprocessGroupSize;

            this.PreprocessCache = clProvider.CreateFloatMem(
                currentLayer.NonBiasNeuronCount * _aggregationFactor,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );

            _preprocessKernel0 = clProvider.CreateKernel(
                kernelTextProvider.GetPreprocessHiddenKernelZeroSource(PreprocessGroupSize),
                "PreprocessKernel0"
                );

            _preprocessKernel1 = clProvider.CreateKernel(
                kernelTextProvider.GetPreprocessHiddenKernelOneSource(),
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
                .SetKernelArg(3, sizeof(int), _currentLayer.NonBiasNeuronCount)
                .SetKernelArg(4, sizeof(int), _nextLayer.NonBiasNeuronCount)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        Helper.UpTo(_currentLayer.NonBiasNeuronCount, PreprocessGroupSize),
                        Helper.UpTo(_nextLayer.NonBiasNeuronCount, PreprocessGroupSize)
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

                var globalx = Helper.UpTo(_currentLayer.NonBiasNeuronCount, currentLocalGroupSize);
                var globaly = Helper.UpTo(aggregationFactor, currentLocalGroupSize);

                _preprocessKernel1
                    .SetKernelArgMem(0, this.PreprocessCache)
                    .SetKernelArg(1, sizeof(int), _currentLayer.NonBiasNeuronCount)
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
    }
}