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
    public class CSharpConvolutionFullConnectedLayerBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;

        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpConvolutionLayerContainer _currentLayerContainer;
        private readonly ICSharpDeDyAggregator _nextLayerDeDyAggregator;

        private readonly float[] _nablaKernel;
        private readonly float[] _nablaBias;

        private readonly ConvolutionFullConnectedLayerKernel _convolutionFullConnectedLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;

        private readonly float[] _deDz;

        public CSharpConvolutionFullConnectedLayerBackpropagator(
            ILearningAlgorithmConfig config,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpConvolutionLayerContainer currentLayerContainer,
            ICSharpDeDyAggregator nextLayerDeDyAggregator
            )
        {
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
            if (nextLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("nextLayerDeDyAggregator");
            }

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;

            _nablaKernel = new float[_currentLayerContainer.Configuration.WeightCount];
            _nablaBias = new float[_currentLayerContainer.Configuration.BiasCount];

            this._deDz = new float[_currentLayerContainer.Configuration.TotalNeuronCount];

            _convolutionFullConnectedLayerKernel = new ConvolutionFullConnectedLayerKernel(
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

                //var currentState = new ReferencedSquareFloat(
                //    _currentLayerContainer.Configuration.SpatialDimension,
                //    _currentLayerContainer.StateMem,
                //    currentNetStateDeDzShift
                //    );

                var dedz = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension,
                    this._deDz,
                    currentNetStateDeDzShift
                    );

                var kernelShift = fmi * _currentLayerContainer.Configuration.KernelSpatialDimension.Multiplied;
                var biasShift = fmi;

                var nabla = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.KernelSpatialDimension,
                    _nablaKernel,
                    kernelShift
                    );

                if (firstItemInBatch)
                {
                    _convolutionFullConnectedLayerKernel.CalculateOverwrite(
                        fmi * _currentLayerContainer.Configuration.SpatialDimension.Multiplied,
                        currentNet,
                        previousState,
                        dedz,
                        nabla,
                        ref _nablaBias[biasShift],
                        _nextLayerDeDyAggregator.DeDy,
                        learningRate
                        );
                }
                else
                {
                    throw new NotImplementedException();
                }
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