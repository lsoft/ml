using System;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
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
        private readonly ILayer _previousLayer;
        private readonly IConvolutionLayer _currentLayer;

        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerContainer;
        private readonly ICSharpLayerContainer _nextLayerContainer;

        private readonly float[] _nextLayerDeDz;

        private readonly float[] _nablaKernel;
        private readonly float[] _nablaBias;

        private readonly ConvolutionFullConnectedLayerKernel _convolutionFullConnectedLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;

        public float[] DeDz
        {
            get;
            private set;
        }

        public CSharpConvolutionFullConnectedLayerBackpropagator(
            ILearningAlgorithmConfig config,
            ILayer previousLayer,
            IConvolutionLayer currentLayer,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpLayerContainer nextLayerContainer,
            float[] nextLayerDeDz
            )
        {
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (previousLayer == null)
            {
                throw new ArgumentNullException("previousLayer");
            }
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
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
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }

            _config = config;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerContainer = nextLayerContainer;
            _nextLayerDeDz = nextLayerDeDz;

            _nablaKernel = new float[_currentLayer.GetConfiguration().WeightCount];
            _nablaBias = new float[_currentLayer.GetConfiguration().BiasCount];

            this.DeDz = new float[_currentLayer.GetConfiguration().TotalNeuronCount];

            _convolutionFullConnectedLayerKernel = new ConvolutionFullConnectedLayerKernel(
                currentLayer
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
            for (var fmi = 0; fmi < _currentLayer.FeatureMapCount; fmi++)
            {
                var previousState = new ReferencedSquareFloat(
                    _previousLayer.SpatialDimension,
                    _previousLayerContainer.StateMem,
                    0
                    );

                var currentNetStateDeDzShift = fmi * _currentLayer.SpatialDimension.Multiplied;

                var currentNet = new ReferencedSquareFloat(
                    _currentLayer.SpatialDimension,
                    _currentLayerContainer.NetMem,
                    currentNetStateDeDzShift
                    );

                var currentState = new ReferencedSquareFloat(
                    _currentLayer.SpatialDimension,
                    _currentLayerContainer.StateMem,
                    currentNetStateDeDzShift
                    );

                var dedz = new ReferencedSquareFloat(
                    _currentLayer.SpatialDimension,
                    this.DeDz,
                    currentNetStateDeDzShift
                    );

                var kernelShift = fmi * _currentLayer.KernelSpatialDimension.Multiplied;
                var biasShift = fmi;

                var nabla = new ReferencedSquareFloat(
                    _currentLayer.KernelSpatialDimension,
                    _nablaKernel,
                    kernelShift
                    );

                if (firstItemInBatch)
                {
                    _convolutionFullConnectedLayerKernel.CalculateOverwrite(
                        fmi * _currentLayer.SpatialDimension.Multiplied,
                        currentNet,
                        currentState,
                        previousState,
                        dedz,
                        _nextLayerDeDz,
                        nabla,
                        ref _nablaBias[biasShift],
                        learningRate,
                        _nextLayerContainer.WeightMem
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
            for (var fmi = 0; fmi < _currentLayer.FeatureMapCount; fmi++)
            {
                var kernelShift = fmi*_currentLayer.KernelSpatialDimension.Multiplied;
                var biasShift = fmi;

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    _currentLayer.KernelSpatialDimension,
                    _currentLayerContainer.WeightMem,
                    kernelShift,
                    _currentLayerContainer.BiasMem,
                    biasShift
                    );

                var nabla = new ReferencedSquareFloat(
                    _currentLayer.KernelSpatialDimension,
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