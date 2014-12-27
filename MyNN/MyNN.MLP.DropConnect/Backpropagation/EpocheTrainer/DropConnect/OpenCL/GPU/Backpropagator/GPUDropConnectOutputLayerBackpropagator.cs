using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.DesiredValues;
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
    public class GPUDropConnectOutputLayerBackpropagator : IMemLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IMemDesiredValuesContainer _desiredValuesContainer;
        private readonly IDropConnectLayerPropagator _currentLayerPropagator;
        private readonly ILayer _outputLayer;
        private readonly ILayer _preOutputLayer;
        private readonly Kernel _outputKernelIncrement;
        private readonly Kernel _outputKernelOverwrite;
        
        private readonly MemFloat _nablaWeights;
        private readonly Kernel _updateWeightKernel;

        public MemFloat DeDz
        {
            get;
            private set;
        }


        public GPUDropConnectOutputLayerBackpropagator(
            CLProvider clProvider,
            IMLP mlp,
            ILearningAlgorithmConfig config,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IKernelTextProvider kernelTextProvider,
            IMemDesiredValuesContainer desiredValuesContainer,
            IDropConnectLayerPropagator currentLayerPropagator
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
            if (currentLayerPropagator == null)
            {
                throw new ArgumentNullException("currentLayerPropagator");
            }

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _desiredValuesContainer = desiredValuesContainer;
            _currentLayerPropagator = currentLayerPropagator;

            var layerIndex = mlp.Layers.Length - 1;

            _outputLayer = mlp.Layers[layerIndex];
            _preOutputLayer = mlp.Layers[layerIndex - 1];

            _nablaWeights = clProvider.CreateFloatMem(
                (_outputLayer.NonBiasNeuronCount) * _outputLayer.Neurons[0].Weights.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            DeDz = clProvider.CreateFloatMem(
                _outputLayer.NonBiasNeuronCount,
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
        }

        public void Backpropagate(
            int dataCount, 
            float learningRate, 
            bool firstItemInBatch
            )
        {
            const int outputLocalSize = 128;
            uint outputGlobalSize = outputLocalSize * (uint)_outputLayer.NonBiasNeuronCount;

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

                    .SetKernelArgMem(7, _currentLayerPropagator.MaskContainer.MaskMem)

                    .SetKernelArg(8, 4, _preOutputLayer.Neurons.Length)
                    .SetKernelArg(9, 4, _outputLayer.NonBiasNeuronCount)

                    .SetKernelArg(10, 4, learningRate)
                    .SetKernelArg(11, 4, _config.RegularizationFactor)
                    .SetKernelArg(12, 4, (float)(dataCount))

                    .SetKernelArg(13, 4, _currentLayerPropagator.MaskContainer.BitMask)

                    .EnqueueNDRangeKernel(
                        new uint[]
                        {
                            outputGlobalSize
                        }
                        , new uint[]
                        {
                            outputLocalSize
                        }
                    );
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

                    .SetKernelArgMem(7, _currentLayerPropagator.MaskContainer.MaskMem)

                    .SetKernelArg(8, 4, _preOutputLayer.Neurons.Length)
                    .SetKernelArg(9, 4, _outputLayer.NonBiasNeuronCount)

                    .SetKernelArg(10, 4, learningRate)
                    .SetKernelArg(11, 4, _config.RegularizationFactor)
                    .SetKernelArg(12, 4, (float)(dataCount))

                    .SetKernelArg(13, 4, _currentLayerPropagator.MaskContainer.BitMask)

                    .EnqueueNDRangeKernel(
                        new uint[]
                        {
                            outputGlobalSize
                        }
                        , new uint[]
                        {
                            outputLocalSize
                        }
                    );
            }
        }

        public void UpdateWeights()
        {
            var weightMem = _currentLayerContainer.WeightMem;
            var nablaMem = _nablaWeights;

            _updateWeightKernel
                .SetKernelArgMem(0, weightMem)
                .SetKernelArgMem(1, nablaMem)
                .SetKernelArg(2, 4, (float)(_config.BatchSize))
                .SetKernelArg(3, 4, weightMem.Array.Length)
                .EnqueueNDRangeKernel(weightMem.Array.Length)
                ;
        }

    }
}