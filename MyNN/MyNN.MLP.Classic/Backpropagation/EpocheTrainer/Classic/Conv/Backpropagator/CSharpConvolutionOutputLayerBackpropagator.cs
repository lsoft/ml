using System;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ErrorCalculator;
using MyNN.MLP.Convolution.ErrorCalculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Backpropagator
{
    public class CSharpConvolutionOutputLayerBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpConvolutionLayerContainer _currentLayerContainer;
        private readonly ICSharpDesiredValuesContainer _desiredValuesContainer;

        private readonly float[] _nablaKernel;
        private float _nablaBias;

        private readonly OutputLayerKernel _outputLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;

        private readonly float[] _deDz;

        public CSharpConvolutionOutputLayerBackpropagator(
            ILearningAlgorithmConfig config,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpConvolutionLayerContainer currentLayerContainer,
            ICSharpDesiredValuesContainer desiredValuesContainer,
            IErrorCalculator errorCalculator
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
            if (desiredValuesContainer == null)
            {
                throw new ArgumentNullException("desiredValuesContainer");
            }
            if (errorCalculator == null)
            {
                throw new ArgumentNullException("errorCalculator");
            }

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _desiredValuesContainer = desiredValuesContainer;

            if (currentLayerContainer.Configuration.FeatureMapCount != 1)
            {
                throw new NotSupportedException("Этот бекпропагатор используется только для отладки, и не поддерживает множественные фича-мапы");
            }
            
            ConsoleAmbientContext.Console.WriteWarning(
                "Этот бекпропагатор используется только для отладки, и не поддерживает множественные фича-мапы"
                );


            _nablaKernel = new float[_currentLayerContainer.Configuration.KernelSpatialDimension.Multiplied];
            _nablaBias = 0f;

            this._deDz = new float[_currentLayerContainer.Configuration.SpatialDimension.Multiplied];

            _outputLayerKernel = new OutputLayerKernel(
                _currentLayerContainer.Configuration,
                config,
                errorCalculator
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
            var currentNet = new ReferencedSquareFloat(
                _currentLayerContainer.Configuration.SpatialDimension,
                _currentLayerContainer.NetMem,
                0
                );

            var currentState = new ReferencedSquareFloat(
                _currentLayerContainer.Configuration.SpatialDimension,
                _currentLayerContainer.StateMem,
                0
                );

            var previousState = new ReferencedSquareFloat(
                _previousLayerContainer.Configuration.SpatialDimension,
                _previousLayerContainer.StateMem,
                0
                );

            var nabla = new ReferencedSquareFloat(
                _currentLayerContainer.Configuration.KernelSpatialDimension,
                _nablaKernel,
                0
                );

            var dedz = new ReferencedSquareFloat(
                _currentLayerContainer.Configuration.SpatialDimension,
                this._deDz,
                0
                );

            if (firstItemInBatch)
            {
                _outputLayerKernel.CalculateOverwrite(
                    currentNet,
                    currentState,
                    _desiredValuesContainer.DesiredOutput,
                    previousState,
                    nabla,
                    ref _nablaBias,
                    learningRate,
                    dedz
                    );
            }
            else
            {
                _outputLayerKernel.CalculateIncrement(
                    currentNet,
                    currentState,
                    _desiredValuesContainer.DesiredOutput,
                    previousState,
                    nabla,
                    ref _nablaBias,
                    learningRate,
                    dedz
                    );
            }
        }

        public void UpdateWeights()
        {
            var kernelBiasContainer = new ReferencedKernelBiasContainer(
                _currentLayerContainer.Configuration.KernelSpatialDimension,
                _currentLayerContainer.WeightMem,
                0,
                _currentLayerContainer.BiasMem,
                0
                );

            var nabla = new ReferencedSquareFloat(
                _currentLayerContainer.Configuration.KernelSpatialDimension,
                _nablaKernel,
                0
                );

            _updateWeightKernel.UpdateWeigths(
                kernelBiasContainer,
                nabla,
                _nablaBias,
                (float) (_config.BatchSize)
                );
        }

    }
}