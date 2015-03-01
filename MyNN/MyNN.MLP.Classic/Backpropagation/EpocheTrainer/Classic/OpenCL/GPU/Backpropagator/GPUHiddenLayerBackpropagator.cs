using System;
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

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU.Backpropagator
{
    public class GPUHiddenLayerBackpropagator : IMemLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly bool _needToCalculateDeDy;
        //private readonly int _layerIndex;
        //private readonly ILayer _previousLayer;
        //private readonly ILayer _currentLayer;
        
        private readonly IMemLayerContainer _previousLayerContainer;
        private readonly IMemLayerContainer _currentLayerContainer;

        private readonly Kernel _hiddenKernelOverwrite;
        private readonly Kernel _hiddenKernelIncrement;

        private readonly MemFloat _nablaWeights;
        private readonly MemFloat _nablaBias;

        private readonly Kernel _updateWeightKernel;

        private readonly IOpenCLDeDyAggregator _nextLayerDeDyAggregator;
        private readonly IOpenCLDeDyAggregator _currentLayerDeDyAggregator;


        public GPUHiddenLayerBackpropagator(
            CLProvider clProvider,
            ILearningAlgorithmConfig config,
            bool needToCalculateDeDy,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IMemLayerContainer nextLayerContainer,
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
            if (nextLayerContainer == null)
            {
                throw new ArgumentNullException("nextLayerContainer");
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
                "UpdateWeightKernel"
                );

            _hiddenKernelIncrement = clProvider.CreateKernel(
                kernelTextProvider.GetIncrementCalculationKernelsSource(
                    currentLayerContainer.Configuration
                    ),
                "HiddenLayerTrain"
                );

            _hiddenKernelOverwrite = clProvider.CreateKernel(
                kernelTextProvider.GetOverwriteCalculationKernelsSource(
                    currentLayerContainer.Configuration
                    ),
                "HiddenLayerTrain"
                );
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
            const uint HiddenLocalGroupSize = 64;
            uint HiddenGlobalGroupSize =
                (uint)_currentLayerContainer.Configuration.TotalNeuronCount * HiddenLocalGroupSize
                ;

            if (firstItemInBatch)
            {
                _hiddenKernelOverwrite
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerDeDyAggregator.DeDz)
                    .SetKernelArgMem(3, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(4, _nablaWeights)
                    .SetKernelArg(5, 4, _previousLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(6, 4, _currentLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(7, 4, learningRate)
                    .SetKernelArg(8, 4, _config.RegularizationFactor)
                    .SetKernelArg(9, 4, (float) (dataCount))
                    .SetKernelArgLocalMem(10, sizeof (float)*HiddenLocalGroupSize)
                    .SetKernelArgMem(11, _nextLayerDeDyAggregator.DeDy)
                    .SetKernelArgMem(12, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(13, _nablaBias)
                    .EnqueueNDRangeKernel(
                        new[]
                        {
                            HiddenGlobalGroupSize
                        },
                        new[]
                        {
                            HiddenLocalGroupSize
                        })
                    ;
            }
            else
            {
                _hiddenKernelIncrement
                    .SetKernelArgMem(0, _currentLayerContainer.NetMem)
                    .SetKernelArgMem(1, _previousLayerContainer.StateMem)
                    .SetKernelArgMem(2, _currentLayerDeDyAggregator.DeDz)
                    .SetKernelArgMem(3, _currentLayerContainer.WeightMem)
                    .SetKernelArgMem(4, _nablaWeights)
                    .SetKernelArg(5, 4, _previousLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(6, 4, _currentLayerContainer.Configuration.TotalNeuronCount)
                    .SetKernelArg(7, 4, learningRate)
                    .SetKernelArg(8, 4, _config.RegularizationFactor)
                    .SetKernelArg(9, 4, (float) (dataCount))
                    .SetKernelArgLocalMem(10, sizeof (float)*HiddenLocalGroupSize)
                    .SetKernelArgMem(11, _nextLayerDeDyAggregator.DeDy)
                    .SetKernelArgMem(12, _currentLayerContainer.BiasMem)
                    .SetKernelArgMem(13, _nablaBias)
                    .EnqueueNDRangeKernel(
                        new[]
                        {
                            HiddenGlobalGroupSize
                        },
                        new[]
                        {
                            HiddenLocalGroupSize
                        })
                    ;
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