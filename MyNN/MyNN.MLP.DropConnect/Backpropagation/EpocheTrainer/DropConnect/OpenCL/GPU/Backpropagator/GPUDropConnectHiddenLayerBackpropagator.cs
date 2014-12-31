using System;
using MyNN.Common;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.NextLayerAggregator;
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
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;
        
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IDropConnectLayerPropagator _currentLayerPropagator;

        private readonly Kernel _hiddenKernelOverwrite;
        private readonly Kernel _hiddenKernelIncrement;

        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _nablaBias;

        private readonly MemFloat _currentDeDz;

        private readonly Kernel _updateWeightKernel;
        private readonly INextLayerAggregator _aggregator;

        public MemFloat DeDz
        {
            get
            {
                return
                    _currentDeDz;
            }
        }

        public GPUDropConnectHiddenLayerBackpropagator(
            CLProvider clProvider,
            IMLP mlp,
            ILearningAlgorithmConfig config,
            int layerIndex,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IMemLayerContainer nextLayerContainer,
            IKernelTextProvider kernelTextProvider,
            MemFloat nextLayerDeDz,
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
            if (currentLayerPropagator == null)
            {
                throw new ArgumentNullException("currentLayerPropagator");
            }

            var previousLayer = mlp.Layers[layerIndex - 1];
            var currentLayer = mlp.Layers[layerIndex];
            var nextLayer = mlp.Layers[layerIndex + 1];

            _config = config;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _currentLayerPropagator = currentLayerPropagator;

            _nablaWeights = clProvider.CreateFloatMem(
                currentLayer.TotalNeuronCount * previousLayer.TotalNeuronCount, //currentLayer.Neurons[0].Weights.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _nablaBias = clProvider.CreateFloatMem(
                currentLayer.TotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _currentDeDz = clProvider.CreateFloatMem(
                currentLayer.TotalNeuronCount,
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

            _aggregator = new CLNextLayerAggregator(
                clProvider,
                currentLayer.GetConfiguration(),
                nextLayer.GetConfiguration(),
                nextLayerDeDz,
                nextLayerContainer
                );
        }

        public void Prepare()
        {
            _nablaWeights.Write(BlockModeEnum.NonBlocking);
            _nablaBias.Write(BlockModeEnum.NonBlocking);

            _aggregator.ClearAndWrite();
        }

        public void Backpropagate(
            int dataCount,
            float learningRate,
            bool firstItemInBatch
            )
        {
            _aggregator.Aggregate();

            {
                const uint hiddenLocalSize = 256;
                uint hiddenGlobalSize =
                    hiddenLocalSize*
                    (uint)_currentLayer.TotalNeuronCount;

                if (firstItemInBatch)
                {
                    _hiddenKernelOverwrite
                        .SetKernelArgMem(0, _currentLayerContainer.NetMem)

                        .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                        .SetKernelArgMem(2, this.DeDz)
                        .SetKernelArgMem(3, _currentLayerContainer.WeightMem)

                        .SetKernelArgMem(4, _nablaWeights)

                        .SetKernelArgMem(5, _currentLayerPropagator.MaskContainer.MaskMem)

                        .SetKernelArg(6, 4, _previousLayer.TotalNeuronCount)
                        .SetKernelArg(7, 4, _currentLayer.TotalNeuronCount)

                        .SetKernelArg(8, 4, learningRate)
                        .SetKernelArg(9, 4, _config.RegularizationFactor)
                        .SetKernelArg(10, 4, (float) (dataCount))

                        .SetKernelArg(11, 4, _currentLayerPropagator.MaskContainer.BitMask)

                        .SetKernelArgLocalMem(12, hiddenLocalSize*sizeof (float))

                        .SetKernelArgMem(13, _aggregator.PreprocessCache)

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
                        .SetKernelArgMem(2, this.DeDz)
                        .SetKernelArgMem(3, _currentLayerContainer.WeightMem)

                        .SetKernelArgMem(4, _nablaWeights)

                        .SetKernelArgMem(5, _currentLayerPropagator.MaskContainer.MaskMem)

                        .SetKernelArg(6, 4, _previousLayer.TotalNeuronCount)
                        .SetKernelArg(7, 4, _currentLayer.TotalNeuronCount)

                        .SetKernelArg(8, 4, learningRate)
                        .SetKernelArg(9, 4, _config.RegularizationFactor)
                        .SetKernelArg(10, 4, (float) (dataCount))

                        .SetKernelArg(11, 4, _currentLayerPropagator.MaskContainer.BitMask)

                        .SetKernelArgLocalMem(12, hiddenLocalSize*sizeof (float))

                        .SetKernelArgMem(13, _aggregator.PreprocessCache)

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