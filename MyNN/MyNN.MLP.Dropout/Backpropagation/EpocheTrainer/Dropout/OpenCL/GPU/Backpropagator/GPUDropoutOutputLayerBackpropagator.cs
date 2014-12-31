using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
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

namespace MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU.Backpropagator
{
    public class GPUDropoutOutputLayerBackpropagator : IMemLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IMemDesiredValuesContainer _desiredValuesContainer;
        private readonly ILayer _outputLayer;
        private readonly ILayer _preOutputLayer;
        private readonly Kernel _outputKernelIncrement;
        private readonly Kernel _outputKernelOverwrite;
        
        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _nablaBias;

        private readonly Kernel _updateWeightKernel;

        public MemFloat DeDz
        {
            get;
            private set;
        }


        public GPUDropoutOutputLayerBackpropagator(
            CLProvider clProvider,
            IMLP mlp,
            ILearningAlgorithmConfig config,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IKernelTextProvider kernelTextProvider,
            IMemDesiredValuesContainer desiredValuesContainer
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

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _desiredValuesContainer = desiredValuesContainer;

            var layerIndex = mlp.Layers.Length - 1;

            _outputLayer = mlp.Layers[layerIndex];
            _preOutputLayer = mlp.Layers[layerIndex - 1];

            _nablaWeights = clProvider.CreateFloatMem(
                (_outputLayer.TotalNeuronCount) * _preOutputLayer.TotalNeuronCount, //_outputLayer.Neurons[0].Weights.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _nablaBias = clProvider.CreateFloatMem(
                _outputLayer.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            DeDz = clProvider.CreateFloatMem(
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
        }

        public void Backpropagate(
            int dataCount, 
            float learningRate, 
            bool firstItemInBatch
            )
        {
            const uint OutputLocalGroupSize = 128;
            uint OutputGlobalGroupSize =
                (uint)_outputLayer.TotalNeuronCount * OutputLocalGroupSize;

            if (firstItemInBatch)
            {
                _outputKernelOverwrite
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, this.DeDz)
                    .SetKernelArgMem(4, _desiredValuesContainer.DesiredOutput)
                    .SetKernelArgMem(5, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(6, _nablaWeights)
                    .SetKernelArg(7, 4, _preOutputLayer.TotalNeuronCount)
                    .SetKernelArg(8, 4, _outputLayer.TotalNeuronCount)
                    .SetKernelArg(9, 4, learningRate)
                    .SetKernelArg(10, 4, _config.RegularizationFactor)
                    .SetKernelArg(11, 4, (float)(dataCount))
                    .SetKernelArgMem(12, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(13, _nablaBias)
                    .EnqueueNDRangeKernel(
                        new[]
                        {
                            OutputGlobalGroupSize
                        },
                        new[]
                        {
                            OutputLocalGroupSize
                        })
                    ;
            }
            else
            {
                _outputKernelIncrement
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerContainer.StateMem)
                    .SetKernelArgMem(3, this.DeDz)
                    .SetKernelArgMem(4, _desiredValuesContainer.DesiredOutput)
                    .SetKernelArgMem(5, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(6, _nablaWeights)
                    .SetKernelArg(7, 4, _preOutputLayer.TotalNeuronCount)
                    .SetKernelArg(8, 4, _outputLayer.TotalNeuronCount)
                    .SetKernelArg(9, 4, learningRate)
                    .SetKernelArg(10, 4, _config.RegularizationFactor)
                    .SetKernelArg(11, 4, (float)(dataCount))
                    .SetKernelArgMem(12, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(13, _nablaBias)
                    .EnqueueNDRangeKernel(
                        new[]
                        {
                            OutputGlobalGroupSize
                        },
                        new[]
                        {
                            OutputLocalGroupSize
                        })
                    ;
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