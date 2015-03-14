using System;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Backpropagator
{
    public class CSharpConvolutionPoolingLayerBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly IAvgPoolingLayerConfiguration _nextLayerConfiguration;

        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpConvolutionLayerContainer _currentLayerContainer;
        private readonly ICSharpDeDyAggregator _nextLayerDeDyAggregator;
        private readonly ICSharpDeDyAggregator _currentLayerDeDyAggregator;
        private readonly bool _needToCalculateDeDy;

        private readonly float[] _nablaKernel;
        private readonly float[] _nablaBias;

        private readonly ConvolutionPoolingLayerKernel _calculateNablaKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;

        public CSharpConvolutionPoolingLayerBackpropagator(
            ILearningAlgorithmConfig config,
            IAvgPoolingLayerConfiguration nextLayerConfiguration,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpConvolutionLayerContainer currentLayerContainer,
            ICSharpDeDyAggregator nextLayerDeDyAggregator,
            ICSharpDeDyAggregator currentLayerDeDyAggregator,
            bool needToCalculateDeDy
            )
        {
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (nextLayerConfiguration == null)
            {
                throw new ArgumentNullException("nextLayerConfiguration");
            }
            if (previousLayerContainer == null)
            {
                throw new ArgumentNullException("previousLayerContainer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
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
            _nextLayerConfiguration = nextLayerConfiguration;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;
            _needToCalculateDeDy = needToCalculateDeDy;

            _nablaKernel = new float[_currentLayerContainer.Configuration.WeightCount];
            _nablaBias = new float[_currentLayerContainer.Configuration.BiasCount];

            _calculateNablaKernel = new ConvolutionPoolingLayerKernel(
                _currentLayerContainer.Configuration
                );

            _updateWeightKernel = new UpdateWeightKernel();
        }

        public void Prepare()
        {
            //nothing to do
        }

        public void Backpropagate(
            int dataCount,
            float learningRate,
            bool firstItemInBatch
            )
        {
            for (var fmi = 0; fmi < _currentLayerContainer.Configuration.FeatureMapCount; fmi++)
            {
                var previousState = new ReferencedSquareFloat(
                    _previousLayerContainer.Configuration.SpatialDimension,
                    _previousLayerContainer.StateMem,
                    0
                    );

                var currentNetStateDeDzShift = fmi * _currentLayerContainer.Configuration.SpatialDimension.Multiplied;

                var currentNet = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension,
                    _currentLayerContainer.NetMem,
                    currentNetStateDeDzShift
                    );

                var currentLayerDeDz = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension,
                    _currentLayerDeDyAggregator.DeDz,
                    currentNetStateDeDzShift
                    );

                var kernelShift = fmi * _currentLayerContainer.Configuration.KernelSpatialDimension.Multiplied;
                var biasShift = fmi;

                var nabla = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.KernelSpatialDimension,
                    _nablaKernel,
                    kernelShift
                    );

                var nldedzShift = fmi * _nextLayerConfiguration.SpatialDimension.Multiplied;

                var nextLayerDeDy = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension, //именно текущий слой!
                    _nextLayerDeDyAggregator.DeDy,
                    nldedzShift
                    );

                if (firstItemInBatch)
                {
                    _calculateNablaKernel.CalculateOverwrite(
                        currentNet,
                        currentLayerDeDz,
                        previousState,
                        nabla,
                        ref _nablaBias[biasShift],
                        nextLayerDeDy,
                        learningRate
                        );
                }
                else
                {
                    throw new NotImplementedException();
                }
            }

            if (_needToCalculateDeDy)
            {
                _currentLayerDeDyAggregator.Aggregate();
            }
        }

        public void UpdateWeights()
        {
            for (var fmi = 0; fmi < _currentLayerContainer.Configuration.FeatureMapCount; fmi++)
            {
                var kernelShift = fmi * _currentLayerContainer.Configuration.KernelSpatialDimension.Multiplied;
                var biasShift = fmi;

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    _currentLayerContainer.Configuration.KernelSpatialDimension,
                    _currentLayerContainer.WeightMem,
                    kernelShift,
                    _currentLayerContainer.BiasMem,
                    biasShift
                    );

                var nabla = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.KernelSpatialDimension,
                    _nablaKernel,
                    kernelShift
                    );

                _updateWeightKernel.UpdateWeigths(
                    kernelBiasContainer,
                    nabla,
                    _nablaBias[biasShift],
                    (float) (_config.BatchSize)
                    );
            }
        }

    }
}