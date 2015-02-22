using System;
using MyNN.MLP.Convolution.Activator;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.ForwardPropagation.CSharp
{
    public class CSharpAvgPooling_ConvolutionLayerPropagator : ILayerPropagator
    {
        private readonly IAvgPoolingLayer _previousLayer;
        private readonly IConvolutionLayer _currentLayer;
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerMemContainer;
        private readonly ICSharpConvolutionCalculator _convolutionCalculator;
        private readonly IFunctionActivator _functionActivator;

        public CSharpAvgPooling_ConvolutionLayerPropagator(
            IAvgPoolingLayer previousLayer,
            IConvolutionLayer currentLayer,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerMemContainer,
            ICSharpConvolutionCalculator convolutionCalculator,
            IFunctionActivator functionActivator
            )
        {
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
            if (currentLayerMemContainer == null)
            {
                throw new ArgumentNullException("currentLayerMemContainer");
            }
            if (convolutionCalculator == null)
            {
                throw new ArgumentNullException("convolutionCalculator");
            }
            if (functionActivator == null)
            {
                throw new ArgumentNullException("functionActivator");
            }

            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerMemContainer = currentLayerMemContainer;
            _convolutionCalculator = convolutionCalculator;
            _functionActivator = functionActivator;
        }

        public void ComputeLayer()
        {
            for (var currentFmi = 0; currentFmi < _currentLayer.FeatureMapCount; currentFmi++)
            {
                var kernelShift = currentFmi * _currentLayer.KernelSpatialDimension.Multiplied;
                var biasShift = currentFmi;

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    _currentLayer.KernelSpatialDimension,
                    _currentLayerMemContainer.WeightMem,
                    kernelShift,
                    _currentLayerMemContainer.BiasMem,
                    biasShift
                    );

                var currentNetStateShift = currentFmi * _currentLayer.SpatialDimension.Multiplied;

                var currentNet = new ReferencedSquareFloat(
                    _currentLayer.SpatialDimension,
                    _currentLayerMemContainer.NetMem,
                    currentNetStateShift
                    );

                var currentState = new ReferencedSquareFloat(
                    _currentLayer.SpatialDimension,
                    _currentLayerMemContainer.StateMem,
                    currentNetStateShift
                    );

                for (var previousFmi = 0; previousFmi < _previousLayer.FeatureMapCount; previousFmi++)
                {
                    var previousShift = previousFmi * _previousLayer.SpatialDimension.Multiplied;

                    var previousState = new ReferencedSquareFloat(
                        _previousLayer.SpatialDimension,
                        _previousLayerContainer.StateMem,
                        previousShift
                        );

                    if (previousFmi == 0)
                    {
                        _convolutionCalculator.CalculateConvolutionWithOverwrite(
                            kernelBiasContainer,
                            previousState,
                            currentNet
                            );
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
                    _currentLayer.LayerActivationFunction,
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