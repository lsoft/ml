using System;
using MyNN.Common;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
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
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;
        private readonly ILayer _nextLayer;
        
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IMemLayerContainer _nextLayerContainer;
        private readonly MemFloat _nextLayerDeDz;

        private readonly Kernel _hiddenKernelOverwrite;
        private readonly Kernel _hiddenKernelIncrement;

        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _currentDeDz;
        
        private readonly Kernel _updateWeightKernel;

        public MemFloat DeDz
        {
            get
            {
                return
                    _currentDeDz;
            }
        }

        public CPUHiddenLayerBackpropagator(
            CLProvider clProvider,
            IMLP mlp,
            ILearningAlgorithmConfig config,
            int layerIndex,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IMemLayerContainer nextLayerContainer,
            IKernelTextProvider kernelTextProvider,
            MemFloat nextLayerDeDz
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
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
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }

            var previousLayer = mlp.Layers[layerIndex - 1];
            var currentLayer = mlp.Layers[layerIndex];
            var nextLayer = mlp.Layers[layerIndex + 1];

            _config = config;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _nextLayer = nextLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerContainer = nextLayerContainer;
            _nextLayerDeDz = nextLayerDeDz;

            _nablaWeights = clProvider.CreateFloatMem(
                currentLayer.NonBiasNeuronCount * currentLayer.Neurons[0].Weights.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _currentDeDz = clProvider.CreateFloatMem(
                currentLayer.NonBiasNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _updateWeightKernel = clProvider.CreateKernel(
                kernelTextProvider.UpdateWeightKernelSource,
                "UpdateWeightKernel");

            _hiddenKernelIncrement = clProvider.CreateKernel(
                kernelTextProvider.GetIncrementCalculationKernelsSource(layerIndex),
                "HiddenLayerTrain");

            _hiddenKernelOverwrite = clProvider.CreateKernel(
                kernelTextProvider.GetOverwriteCalculationKernelsSource(layerIndex),
                "HiddenLayerTrain");

        }

        public void Prepare()
        {
            _nablaWeights.Write(BlockModeEnum.NonBlocking);
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
                    .SetKernelArgMem(3, this.DeDz)
                    .SetKernelArgMem(4, this._nextLayerDeDz)
                    .SetKernelArgMem(5, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(6, _nextLayerContainer.WeightMem)
                    .SetKernelArgMem(7, _nablaWeights)
                    .SetKernelArg(8, 4, _previousLayer.Neurons.Length / 4)
                    .SetKernelArg(9, 4, _previousLayer.Neurons.Length - (_previousLayer.Neurons.Length % 4))
                    .SetKernelArg(10, 4, _previousLayer.Neurons.Length)
                    .SetKernelArg(11, 4, _currentLayer.NonBiasNeuronCount)
                    .SetKernelArg(12, 4, _nextLayer.NonBiasNeuronCount)
                    .SetKernelArg(13, 4, learningRate)
                    .SetKernelArg(14, 4, _config.RegularizationFactor)
                    .SetKernelArg(15, 4, (float)(dataCount))
                    .EnqueueNDRangeKernel(_currentLayer.NonBiasNeuronCount);
            }
            else
            {
                _hiddenKernelIncrement
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, this.DeDz)
                    .SetKernelArgMem(4, this._nextLayerDeDz)
                    .SetKernelArgMem(5, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(6, _nextLayerContainer.WeightMem)
                    .SetKernelArgMem(7, _nablaWeights)
                    .SetKernelArg(8, 4, _previousLayer.Neurons.Length / 4)
                    .SetKernelArg(9, 4, _previousLayer.Neurons.Length - (_previousLayer.Neurons.Length % 4))
                    .SetKernelArg(10, 4, _previousLayer.Neurons.Length)
                    .SetKernelArg(11, 4, _currentLayer.NonBiasNeuronCount)
                    .SetKernelArg(12, 4, _nextLayer.NonBiasNeuronCount)
                    .SetKernelArg(13, 4, learningRate)
                    .SetKernelArg(14, 4, _config.RegularizationFactor)
                    .SetKernelArg(15, 4, (float)(dataCount))
                    .EnqueueNDRangeKernel(_currentLayer.NonBiasNeuronCount);
            }
        }

        public void UpdateWeights()
        {
            const int perKernelFloats = 1500; //по 1500 флоатов на кернел (должно быть кратно 4м!!!)

            var weightMem = _currentLayerContainer.WeightMem;
            var nablaMem = _nablaWeights;

            var kernelCount = weightMem.Array.Length / perKernelFloats;
            if (weightMem.Array.Length % perKernelFloats > 0)
            {
                kernelCount++;
            }

            _updateWeightKernel
                .SetKernelArgMem(0, weightMem)
                .SetKernelArgMem(1, nablaMem)
                .SetKernelArg(2, 4, weightMem.Array.Length)
                .SetKernelArg(3, 4, perKernelFloats)
                .SetKernelArg(4, 4, (float)(_config.BatchSize))
                .EnqueueNDRangeKernel(kernelCount);
        }

    }
}