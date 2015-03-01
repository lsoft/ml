using System;
using MyNN.Common;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU.Backpropagator
{
    public class CPUHiddenLayerBackpropagator : IMemLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly bool _needToCalculateDeDy;

        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IOpenCLDeDyAggregator _nextLayerDeDyAggregator;
        private readonly IOpenCLDeDyAggregator _currentLayerDeDyAggregator;

        private readonly Kernel _hiddenKernelOverwrite;
        private readonly Kernel _hiddenKernelIncrement;

        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _nablaBias;


        private readonly Kernel _updateWeightKernel;


        public CPUHiddenLayerBackpropagator(
            CLProvider clProvider,
            ILearningAlgorithmConfig config,
            bool needToCalculateDeDy,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IKernelTextProvider kernelTextProvider,
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
            if (kernelTextProvider == null)
            {
                throw new ArgumentNullException("kernelTextProvider");
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
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;

            _nablaWeights = clProvider.CreateFloatMem(
                currentLayerContainer.Configuration.TotalNeuronCount * previousLayerContainer.Configuration.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _nablaBias = clProvider.CreateFloatMem(
                currentLayerContainer.Configuration.TotalNeuronCount,
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
            if (firstItemInBatch)
            {
                _hiddenKernelOverwrite
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, _currentLayerDeDyAggregator.DeDz)
                    .SetKernelArgMem(4, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(5, _nextLayerDeDyAggregator.DeDy)
                    .SetKernelArgMem(6, _nablaWeights)
                    .SetKernelArg(7, 4, _previousLayerContainer.Configuration.TotalNeuronCount / 4)
                    .SetKernelArg(8, 4, _previousLayerContainer.Configuration.TotalNeuronCount - (_previousLayerContainer.Configuration.TotalNeuronCount % 4))
                    .SetKernelArg(9, 4, _previousLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(10, 4, _currentLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(11, 4, learningRate)
                    .SetKernelArg(12, 4, _config.RegularizationFactor)
                    .SetKernelArg(13, 4, (float)(dataCount))
                    .SetKernelArgMem(14, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(15, _nablaBias)
                    .EnqueueNDRangeKernel(_currentLayerContainer.Configuration.TotalNeuronCount);
            }
            else
            {
                _hiddenKernelIncrement
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, _currentLayerDeDyAggregator.DeDz)
                    .SetKernelArgMem(4, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(5, _nextLayerDeDyAggregator.DeDy)
                    .SetKernelArgMem(6, _nablaWeights)
                    .SetKernelArg(7, 4, _previousLayerContainer.Configuration.TotalNeuronCount / 4)
                    .SetKernelArg(8, 4, _previousLayerContainer.Configuration.TotalNeuronCount - (_previousLayerContainer.Configuration.TotalNeuronCount % 4))
                    .SetKernelArg(9, 4, _previousLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(10, 4, _currentLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(11, 4, learningRate)
                    .SetKernelArg(12, 4, _config.RegularizationFactor)
                    .SetKernelArg(13, 4, (float)(dataCount))
                    .SetKernelArgMem(14, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(15, _nablaBias)
                    .EnqueueNDRangeKernel(_currentLayerContainer.Configuration.TotalNeuronCount);
            }

            if (_needToCalculateDeDy)
            {
                _currentLayerDeDyAggregator.Aggregate();
            }
        }

        public void UpdateWeights()
        {
            const int perKernelFloats = 1500; //по 1500 флоатов на кернел (должно быть кратно 4м!!!)

            var weightMem = _currentLayerContainer.WeightMem;
            var nablaMem = _nablaWeights;

            var biasMem = _currentLayerContainer.BiasMem;
            var nablaBias = _nablaBias;

            var kernelCount = weightMem.Array.Length / perKernelFloats;
            if (weightMem.Array.Length % perKernelFloats > 0)
            {
                kernelCount++;
            }

            _updateWeightKernel
                .SetKernelArgMem(0, weightMem)
                .SetKernelArgMem(1, nablaMem)
                .SetKernelArg(2, sizeof(int), weightMem.Array.Length)
                .SetKernelArg(3, sizeof(int), perKernelFloats)
                .SetKernelArg(4, sizeof(float), (float)(_config.BatchSize))
                .SetKernelArgMem(5, biasMem)
                .SetKernelArgMem(6, nablaBias)
                .SetKernelArg(7, sizeof(int), biasMem.Array.Length)
                .EnqueueNDRangeKernel(kernelCount);
        }

    }
}