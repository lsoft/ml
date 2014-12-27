using System;
using MyNN.Common;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
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
        private const int PreprocessGroupSize = 16;

        private readonly ILearningAlgorithmConfig _config;
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;
        private readonly ILayer _nextLayer;
        
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;
        private readonly IMemLayerContainer _nextLayerContainer;
        private readonly MemFloat _nextLayerDeDz;
        private readonly IDropConnectLayerPropagator _currentLayerPropagator;

        private readonly Kernel _hiddenKernelOverwrite;
        private readonly Kernel _hiddenKernelIncrement;

        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _currentDeDz;
        
        private readonly int _aggregationFactor;
        private readonly MemFloat _preprocessCache;

        private readonly Kernel _preprocessKernel0;
        private readonly Kernel _preprocessKernel1;
        private readonly Kernel _updateWeightKernel;

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
            _nextLayer = nextLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerContainer = nextLayerContainer;
            _nextLayerDeDz = nextLayerDeDz;
            _currentLayerPropagator = currentLayerPropagator;

            _nablaWeights = clProvider.CreateFloatMem(
                currentLayer.NonBiasNeuronCount * currentLayer.Neurons[0].Weights.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _currentDeDz = clProvider.CreateFloatMem(
                currentLayer.NonBiasNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            var aggregationFactor = Helper.UpTo(nextLayer.NonBiasNeuronCount, PreprocessGroupSize) / PreprocessGroupSize;

            _aggregationFactor = aggregationFactor;

            _preprocessCache = clProvider.CreateFloatMem(
                currentLayer.NonBiasNeuronCount * aggregationFactor,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite
                );


            _updateWeightKernel = clProvider.CreateKernel(
                kernelTextProvider.UpdateWeightKernelSource,
                "UpdateWeightKernel");

            _preprocessKernel0 = clProvider.CreateKernel(
                kernelTextProvider.GetPreprocessHiddenKernelZeroSource(PreprocessGroupSize),
                "PreprocessKernel0"
                );

            _preprocessKernel1 = clProvider.CreateKernel(
                kernelTextProvider.GetPreprocessHiddenKernelOneSource(),
                "PreprocessKernel1"
                );

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

            _preprocessCache.Array.Clear();
            _preprocessCache.Write(BlockModeEnum.NonBlocking);
        }

        public void Backpropagate(
            int dataCount,
            float learningRate,
            bool firstItemInBatch
            )
        {
            {
                _preprocessKernel0
                    .SetKernelArgMem(0, _nextLayerDeDz)
                    .SetKernelArgMem(1, _nextLayerContainer.WeightMem)
                    .SetKernelArgMem(2, _preprocessCache)
                    .SetKernelArg(3, sizeof (int), _currentLayer.NonBiasNeuronCount)
                    .SetKernelArg(4, sizeof (int), _nextLayer.NonBiasNeuronCount)
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

                for (;;)
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
                        .SetKernelArgMem(0, _preprocessCache)
                        .SetKernelArg(1, sizeof (int), _currentLayer.NonBiasNeuronCount)
                        .SetKernelArg(2, sizeof (int), aggregationFactor)
                        .SetKernelArg(3, sizeof (int), currentLocalGroupSize)
                        .SetKernelArg(4, sizeof (int), currentLocalGroupSize)
                        .SetKernelArgLocalMem(5, sizeof (float)*currentLocalGroupSize*currentLocalGroupSize)
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

                    aggregationFactor = Helper.UpTo(aggregationFactor, currentLocalGroupSize)/currentLocalGroupSize;

                }
            }

            {
                const uint hiddenLocalSize = 256;
                uint hiddenGlobalSize =
                    hiddenLocalSize*
                    (uint) _currentLayer.NonBiasNeuronCount;

                if (firstItemInBatch)
                {
                    _hiddenKernelOverwrite
                        .SetKernelArgMem(0, _currentLayerContainer.NetMem)

                        .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                        .SetKernelArgMem(2, this.DeDz)
                        .SetKernelArgMem(3, _currentLayerContainer.WeightMem)

                        .SetKernelArgMem(4, _nablaWeights)

                        .SetKernelArgMem(5, _currentLayerPropagator.MaskContainer.MaskMem)

                        .SetKernelArg(6, 4, _previousLayer.Neurons.Length)
                        .SetKernelArg(7, 4, _currentLayer.NonBiasNeuronCount)

                        .SetKernelArg(8, 4, learningRate)
                        .SetKernelArg(9, 4, _config.RegularizationFactor)
                        .SetKernelArg(10, 4, (float) (dataCount))

                        .SetKernelArg(11, 4, _currentLayerPropagator.MaskContainer.BitMask)

                        .SetKernelArgLocalMem(12, hiddenLocalSize*sizeof (float))

                        .SetKernelArgMem(13, _preprocessCache)

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

                        .SetKernelArg(6, 4, _previousLayer.Neurons.Length)
                        .SetKernelArg(7, 4, _currentLayer.NonBiasNeuronCount)

                        .SetKernelArg(8, 4, learningRate)
                        .SetKernelArg(9, 4, _config.RegularizationFactor)
                        .SetKernelArg(10, 4, (float) (dataCount))

                        .SetKernelArg(11, 4, _currentLayerPropagator.MaskContainer.BitMask)

                        .SetKernelArgLocalMem(12, hiddenLocalSize*sizeof (float))

                        .SetKernelArgMem(13, _preprocessCache)

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