using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DesiredValues;
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
    public class CPUOutputLayerBackpropagator : IMemLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IMemDesiredValuesContainer _desiredValuesContainer;
        private readonly IOpenCLDeDyAggregator _deDyAggregator;
        private readonly ILayer _outputLayer;
        private readonly ILayer _preOutputLayer;
        private readonly Kernel _outputKernelIncrement;
        private readonly Kernel _outputKernelOverwrite;
        
        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _nablaBias;

        private readonly Kernel _updateWeightKernel;

        public MemFloat DeDz
        {
            get
            {
                throw new InvalidOperationException();
            }
        }


        public CPUOutputLayerBackpropagator(
            CLProvider clProvider,
            IMLP mlp,
            ILearningAlgorithmConfig config,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IKernelTextProvider kernelTextProvider,
            IMemDesiredValuesContainer desiredValuesContainer,
            IOpenCLDeDyAggregator deDyAggregator
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
            if (kernelTextProvider == null)
            {
                throw new ArgumentNullException("kernelTextProvider");
            }
            if (desiredValuesContainer == null)
            {
                throw new ArgumentNullException("desiredValuesContainer");
            }
            if (deDyAggregator == null)
            {
                throw new ArgumentNullException("deDyAggregator");
            }

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _desiredValuesContainer = desiredValuesContainer;
            _deDyAggregator = deDyAggregator;

            var layerIndex = mlp.Layers.Length - 1;

            _outputLayer = mlp.Layers[layerIndex];
            _preOutputLayer = mlp.Layers[layerIndex - 1];

            _nablaWeights = clProvider.CreateFloatMem(
                (_outputLayer.TotalNeuronCount) * _preOutputLayer.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _nablaBias = clProvider.CreateFloatMem(
                _outputLayer.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _updateWeightKernel = clProvider.CreateKernel(
                kernelTextProvider.UpdateWeightKernelSource,
                "UpdateWeightKernel");

            _outputKernelIncrement = clProvider.CreateKernel(
                kernelTextProvider.GetIncrementCalculationKernelsSource(layerIndex),
                "OutputLayerTrain");

            _outputKernelOverwrite = clProvider.CreateKernel(
                kernelTextProvider.GetOverwriteCalculationKernelsSource(layerIndex),
                "OutputLayerTrain");

        }

        public void Prepare()
        {
            _nablaWeights.Write(BlockModeEnum.NonBlocking);
            _nablaBias.Write(BlockModeEnum.NonBlocking);

            this._deDyAggregator.ClearAndWrite();
        }

        public void Backpropagate(
            int dataCount, 
            float learningRate, 
            bool firstItemInBatch
            )
        {
            if (firstItemInBatch)
            {
                _outputKernelOverwrite
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, this._deDyAggregator.DeDz)
                    .SetKernelArgMem(4, _desiredValuesContainer.DesiredOutput)
                    .SetKernelArgMem(5, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(6, _nablaWeights)
                    .SetKernelArg(7, 4, _preOutputLayer.TotalNeuronCount / 4)
                    .SetKernelArg(8, 4, _preOutputLayer.TotalNeuronCount - (_preOutputLayer.TotalNeuronCount % 4))
                    .SetKernelArg(9, 4, _preOutputLayer.TotalNeuronCount)
                    .SetKernelArg(10, 4, _outputLayer.TotalNeuronCount)
                    .SetKernelArg(11, 4, learningRate)
                    .SetKernelArg(12, 4, _config.RegularizationFactor)
                    .SetKernelArg(13, 4, (float)(dataCount))
                    .SetKernelArgMem(14, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(15, _nablaBias)
                    .EnqueueNDRangeKernel(_outputLayer.TotalNeuronCount);
            }
            else
            {
                _outputKernelIncrement
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, this._deDyAggregator.DeDz)
                    .SetKernelArgMem(4, _desiredValuesContainer.DesiredOutput)
                    .SetKernelArgMem(5, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(6, _nablaWeights)
                    .SetKernelArg(7, 4, _preOutputLayer.TotalNeuronCount / 4)
                    .SetKernelArg(8, 4, _preOutputLayer.TotalNeuronCount - (_preOutputLayer.TotalNeuronCount % 4))
                    .SetKernelArg(9, 4, _preOutputLayer.TotalNeuronCount)
                    .SetKernelArg(10, 4, _outputLayer.TotalNeuronCount)
                    .SetKernelArg(11, 4, learningRate)
                    .SetKernelArg(12, 4, _config.RegularizationFactor)
                    .SetKernelArg(13, 4, (float)(dataCount))
                    .SetKernelArgMem(14, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(15, _nablaBias)
                    .EnqueueNDRangeKernel(_outputLayer.TotalNeuronCount);
            }

            this._deDyAggregator.Aggregate();
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