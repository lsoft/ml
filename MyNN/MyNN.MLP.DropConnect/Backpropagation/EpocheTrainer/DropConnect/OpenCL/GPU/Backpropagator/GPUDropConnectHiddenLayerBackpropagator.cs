using System;
using MyNN.Common;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU.Backpropagator
{
    public class GPUDropConnectHiddenLayerBackpropagator : IMemLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly bool _needToCalculateDeDy;

        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IDropConnectLayerPropagator _currentLayerPropagator;
        private readonly IOpenCLDeDyAggregator _nextLayerDeDyAggregator;
        private readonly IOpenCLDeDyAggregator _currentLayerDeDyAggregator;

        private readonly Kernel _hiddenKernelOverwrite;
        private readonly Kernel _hiddenKernelIncrement;

        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _nablaBias;

        private readonly Kernel _updateWeightKernel;

        public GPUDropConnectHiddenLayerBackpropagator(
            CLProvider clProvider,
            ILearningAlgorithmConfig config,
            bool needToCalculateDeDy,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IMemLayerContainer nextLayerContainer,
            IKernelTextProvider kernelTextProvider,
            IDropConnectLayerPropagator currentLayerPropagator,
            IOpenCLDeDyAggregator nextLayerDeDyAggregator,
            IOpenCLDeDyAggregator currentLayerDeDyAggregator
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (previousLayerContainer == null)
            {
                throw new ArgumentNullException("previousLayerContainer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }
            if (nextLayerContainer == null)
            {
                throw new ArgumentNullException("nextLayerContainer");
            }
            if (kernelTextProvider == null)
            {
                throw new ArgumentNullException("kernelTextProvider");
            }
            if (currentLayerPropagator == null)
            {
                throw new ArgumentNullException("currentLayerPropagator");
            }
            if (nextLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("nextLayerDeDyAggregator");
            }
            if (currentLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("currentLayerDeDyAggregator");
            }

            _config = config;
            _needToCalculateDeDy = needToCalculateDeDy;

            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _currentLayerPropagator = currentLayerPropagator;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;

            _nablaWeights = clProvider.CreateFloatMem(
                _currentLayerContainer.Configuration.TotalNeuronCount * _previousLayerContainer.Configuration.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _nablaBias = clProvider.CreateFloatMem(
                _currentLayerContainer.Configuration.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _updateWeightKernel = clProvider.CreateKernel(
                kernelTextProvider.UpdateWeightKernelSource,
                "UpdateWeightKernel");

            _hiddenKernelIncrement = clProvider.CreateKernel(
                kernelTextProvider.GetIncrementCalculationKernelsSource(
                    currentLayerContainer.Configuration
                    ),
                "HiddenLayerTrain");

            _hiddenKernelOverwrite = clProvider.CreateKernel(
                kernelTextProvider.GetOverwriteCalculationKernelsSource(
                    currentLayerContainer.Configuration
                    ),
                "HiddenLayerTrain");
        }

        public void Prepare()
        {
            _nablaWeights.Write(BlockModeEnum.NonBlocking);
            _nablaBias.Write(BlockModeEnum.NonBlocking);

            _currentLayerDeDyAggregator.ClearAndWrite();
        }

        public void Backpropagate(
            int dataCount,
            float learningRate,
            bool firstItemInBatch
            )
        {
            const uint hiddenLocalSize = 256;
            uint hiddenGlobalSize =
                hiddenLocalSize * (uint)_currentLayerContainer.Configuration.TotalNeuronCount;

            if (firstItemInBatch)
            {
                _hiddenKernelOverwrite
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerDeDyAggregator.DeDz)
                    .SetKernelArgMem(3, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(4, _nablaWeights)
                    .SetKernelArgMem(5, _currentLayerPropagator.MaskContainer.MaskMem)
                    .SetKernelArg(6, 4, _previousLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(7, 4, _currentLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(8, 4, learningRate)
                    .SetKernelArg(9, 4, _config.RegularizationFactor)
                    .SetKernelArg(10, 4, (float) (dataCount))
                    .SetKernelArg(11, 4, _currentLayerPropagator.MaskContainer.BitMask)
                    .SetKernelArgLocalMem(12, hiddenLocalSize*sizeof (float))
                    .SetKernelArgMem(13, _nextLayerDeDyAggregator.DeDy)
                    .SetKernelArgMem(14, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(15, _nablaBias)
                    .EnqueueNDRangeKernel(
                        new uint[]
                        {
                            hiddenGlobalSize
                        }
                        , new uint[]
                        {
                            hiddenLocalSize
                        }
                    );
            }
            else
            {
                _hiddenKernelIncrement
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerDeDyAggregator.DeDz)
                    .SetKernelArgMem(3, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(4, _nablaWeights)
                    .SetKernelArgMem(5, _currentLayerPropagator.MaskContainer.MaskMem)
                    .SetKernelArg(6, 4, _previousLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(7, 4, _currentLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(8, 4, learningRate)
                    .SetKernelArg(9, 4, _config.RegularizationFactor)
                    .SetKernelArg(10, 4, (float) (dataCount))
                    .SetKernelArg(11, 4, _currentLayerPropagator.MaskContainer.BitMask)
                    .SetKernelArgLocalMem(12, hiddenLocalSize*sizeof (float))
                    .SetKernelArgMem(13, _nextLayerDeDyAggregator.DeDy)
                    .SetKernelArgMem(14, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(15, _nablaBias)
                    .EnqueueNDRangeKernel(
                        new uint[]
                        {
                            hiddenGlobalSize
                        }
                        , new uint[]
                        {
                            hiddenLocalSize
                        }
                    );
            }

            if (_needToCalculateDeDy)
            {
                _currentLayerDeDyAggregator.Aggregate();
            }
        }

        public void UpdateWeights()
        {
            var weightMem = _currentLayerContainer.WeightMem;
            var nablaMem = _nablaWeights;

            var biasMem = _currentLayerContainer.BiasMem;
            var nablaBias = _nablaBias;

            _updateWeightKernel
                .SetKernelArgMem(0, weightMem)
                .SetKernelArgMem(1, nablaMem)
                .SetKernelArg(2, 4, (float)(_config.BatchSize))
                .SetKernelArg(3, 4, weightMem.Array.Length)
                .SetKernelArgMem(4, biasMem)
                .SetKernelArgMem(5, nablaBias)
                .SetKernelArg(6, sizeof(int), biasMem.Array.Length)
                .EnqueueNDRangeKernel(weightMem.Array.Length)
                ;
        }

    }
}