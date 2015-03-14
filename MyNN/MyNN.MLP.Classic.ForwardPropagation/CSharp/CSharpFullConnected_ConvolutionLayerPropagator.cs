using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Convolution.Activator;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.ForwardPropagation.CSharp
{
    public class CSharpFullConnected_ConvolutionLayerPropagator : ILayerPropagator
    {
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpConvolutionLayerContainer _currentLayerContainer;
        private readonly ICSharpConvolutionCalculator _convolutionCalculator;
        private readonly IFunctionActivator _functionActivator;

        public CSharpFullConnected_ConvolutionLayerPropagator(
            ICSharpLayerContainer previousLayerContainer,
            ICSharpConvolutionLayerContainer currentLayerContainer,
            ICSharpConvolutionCalculator convolutionCalculator,
            IFunctionActivator functionActivator
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

            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _convolutionCalculator = convolutionCalculator;
            _functionActivator = functionActivator;
        }

        public void ComputeLayer()
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

                var previousState = new ReferencedSquareFloat(
                    _previousLayerContainer.Configuration.SpatialDimension,
                    _previousLayerContainer.StateMem,
                    0
                    );

                var currentNetStateShift = fmi * _currentLayerContainer.Configuration.SpatialDimension.Multiplied;

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

                _convolutionCalculator.CalculateConvolutionWithOverwrite(
                    kernelBiasContainer,
                    previousState,
                    currentNet
                    );

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
