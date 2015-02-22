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
        private readonly IFullConnectedLayer _previousLayer;
        private readonly IConvolutionLayer _currentLayer;
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerMemContainer;
        private readonly ICSharpConvolutionCalculator _convolutionCalculator;
        private readonly IFunctionActivator _functionActivator;

        public CSharpFullConnected_ConvolutionLayerPropagator(
            IFullConnectedLayer previousLayer,
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
            for (var fmi = 0; fmi < _currentLayer.FeatureMapCount; fmi++)
            {
                var kernelShift = fmi * _currentLayer.KernelSpatialDimension.Multiplied;
                var biasShift = fmi;

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    _currentLayer.KernelSpatialDimension,
                    _currentLayerMemContainer.WeightMem,
                    kernelShift,
                    _currentLayerMemContainer.BiasMem,
                    biasShift
                    );

                var previousState = new ReferencedSquareFloat(
                    _previousLayer.SpatialDimension,
                    _previousLayerContainer.StateMem,
                    0
                    );

                var currentNetStateShift = fmi * _currentLayer.SpatialDimension.Multiplied;

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

                _convolutionCalculator.CalculateConvolutionWithOverwrite(
                    kernelBiasContainer,
                    previousState,
                    currentNet
                    );

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
