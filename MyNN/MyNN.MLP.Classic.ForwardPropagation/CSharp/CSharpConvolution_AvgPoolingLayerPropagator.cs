using System;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.ForwardPropagation.CSharp
{
    public class CSharpConvolution_AvgPoolingLayerPropagator : ILayerPropagator
    {
        private readonly IConvolutionLayer _previousLayer;
        private readonly IAvgPoolingLayer _currentLayer;
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerMemContainer;
        
        public CSharpConvolution_AvgPoolingLayerPropagator(
            IConvolutionLayer previousLayer,
            IAvgPoolingLayer currentLayer,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerMemContainer
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
            if (previousLayer.FeatureMapCount != currentLayer.FeatureMapCount)
            {
                throw new ArgumentException("previousLayer.FeatureMapCount != currentLayer.FeatureMapCount");
            }
            if (!previousLayer.SpatialDimension.Rescale(currentLayer.ScaleFactor).IsEqual(currentLayer.SpatialDimension))
            {
                throw new ArgumentException("Размеры пред. сверточного слоя и размеры и фактор текущего пулинг слоя не совпадают");
            }

            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerMemContainer = currentLayerMemContainer;
        }

        public void ComputeLayer()
        {
            //downsample
            for (var fmi = 0; fmi < _currentLayer.FeatureMapCount; fmi++)
            {
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

                var previousStateShift = fmi * _previousLayer.SpatialDimension.Multiplied;

                var previousState = new ReferencedSquareFloat(
                    _previousLayer.SpatialDimension,
                    _previousLayerContainer.StateMem,
                    previousStateShift
                    );

                for (var h = 0; h < _currentLayer.SpatialDimension.Height; h++)
                {
                    for (var w = 0; w < _currentLayer.SpatialDimension.Width; w++)
                    {
                        var sum = 0f;

                        for (var hp = h * _currentLayer.InverseScaleFactor; hp < (h * _currentLayer.InverseScaleFactor) + _currentLayer.InverseScaleFactor; hp++)
                        {
                            for (var wp = w * _currentLayer.InverseScaleFactor; wp < (w * _currentLayer.InverseScaleFactor) + _currentLayer.InverseScaleFactor; wp++)
                            {
                                sum += previousState.GetValueFromCoordSafely(wp, hp);
                            }
                        }

                        sum /= _currentLayer.InverseScaleFactor * _currentLayer.InverseScaleFactor;

                        currentNet.SetValueFromCoordSafely(w, h, 0);
                        currentState.SetValueFromCoordSafely(w, h, sum);
                    }
                }
            }
        }

        public void WaitForCalculationFinished()
        {
            //nothing to do
        }
    }
}