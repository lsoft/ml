using System;
using System.Collections.Generic;
using MyNN.MLP.Convolution.Activator;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.Connector;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.ForwardPropagation.CSharp
{
    public class CSharpAvgPooling_ConvolutionLayerPropagator : ILayerPropagator
    {
        private readonly ICSharpAvgPoolingLayerContainer _previousLayerContainer;
        private readonly ICSharpConvolutionLayerContainer _currentLayerContainer;
        private readonly ICSharpConvolutionCalculator _convolutionCalculator;
        private readonly IFunctionActivator _functionActivator;
        private readonly IConnector _connector;

        public CSharpAvgPooling_ConvolutionLayerPropagator(
            ICSharpAvgPoolingLayerContainer previousLayerContainer,
            ICSharpConvolutionLayerContainer currentLayerContainer,
            ICSharpConvolutionCalculator convolutionCalculator,
            IFunctionActivator functionActivator,
            IConnector connector
            )
        {
            if (previousLayerContainer == null)
            {
                throw new ArgumentNullException("previousLayerContainer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }
            if (convolutionCalculator == null)
            {
                throw new ArgumentNullException("convolutionCalculator");
            }
            if (functionActivator == null)
            {
                throw new ArgumentNullException("functionActivator");
            }
            if (connector == null)
            {
                throw new ArgumentNullException("connector");
            }

            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _convolutionCalculator = convolutionCalculator;
            _functionActivator = functionActivator;
            _connector = connector;
        }

        public void ComputeLayer()
        {
            var processedHash = new HashSet<int>();

            for (var currentFmi = 0; currentFmi < _currentLayerContainer.Configuration.FeatureMapCount; currentFmi++)
            {
                var kernelShift = currentFmi * _currentLayerContainer.Configuration.KernelSpatialDimension.Multiplied;
                var biasShift = currentFmi;

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    _currentLayerContainer.Configuration.KernelSpatialDimension,
                    _currentLayerContainer.WeightMem,
                    kernelShift,
                    _currentLayerContainer.BiasMem,
                    biasShift
                    );

                var currentNetStateShift = currentFmi * _currentLayerContainer.Configuration.SpatialDimension.Multiplied;

                var currentNet = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension,
                    _currentLayerContainer.NetMem,
                    currentNetStateShift
                    );

                var currentState = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension,
                    _currentLayerContainer.StateMem,
                    currentNetStateShift
                    );

                foreach (var previousFmi in _connector.GetPreviousFeatureMapIndexes(currentFmi))
                //for (var previousFmi = 0; previousFmi < _previousLayerContainer.Configuration.FeatureMapCount; previousFmi++)
                {
                    var previousShift = previousFmi * _previousLayerContainer.Configuration.SpatialDimension.Multiplied;

                    var previousState = new ReferencedSquareFloat(
                        _previousLayerContainer.Configuration.SpatialDimension,
                        _previousLayerContainer.StateMem,
                        previousShift
                        );

                    if (!processedHash.Contains(currentFmi))
                    {
                        _convolutionCalculator.CalculateConvolutionWithOverwrite(
                            kernelBiasContainer,
                            previousState,
                            currentNet
                            );

                        processedHash.Add(currentFmi);
                    }
                    else
                    {
                        _convolutionCalculator.CalculateConvolutionWithIncrement(
                            kernelBiasContainer,
                            previousState,
                            currentNet
                            );
                    }
                }

                //применяем функцию активации
                _functionActivator.Apply(
                    _currentLayerContainer.Configuration.LayerActivationFunction,
                    currentNet,
                    currentState
                    );
            }
        }

        public void WaitForCalculationFinished()
        {
            //nothing to do
        }
    }
}