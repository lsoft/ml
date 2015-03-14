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
        private readonly ICSharpConvolutionLayerContainer _previousLayerContainer;
        private readonly ICSharpAvgPoolingLayerContainer _currentLayerContainer;
        
        public CSharpConvolution_AvgPoolingLayerPropagator(
            ICSharpConvolutionLayerContainer previousLayerContainer,
            ICSharpAvgPoolingLayerContainer currentLayerContainer
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
            if (previousLayerContainer.Configuration.FeatureMapCount != currentLayerContainer.Configuration.FeatureMapCount)
            {
                throw new ArgumentException("previousLayer.FeatureMapCount != currentLayer.FeatureMapCount");
            }
            if (!previousLayerContainer.Configuration.SpatialDimension.Rescale(currentLayerContainer.Configuration.ScaleFactor).IsEqual(currentLayerContainer.Configuration.SpatialDimension))
            {
                throw new ArgumentException("Размеры пред. сверточного слоя и размеры и фактор текущего пулинг слоя не совпадают");
            }

            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
        }

        public void ComputeLayer()
        {
            //downsample
            for (var fmi = 0; fmi < _currentLayerContainer.Configuration.FeatureMapCount; fmi++)
            {
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

                var previousStateShift = fmi * _previousLayerContainer.Configuration.SpatialDimension.Multiplied;

                var previousState = new ReferencedSquareFloat(
                    _previousLayerContainer.Configuration.SpatialDimension,
                    _previousLayerContainer.StateMem,
                    previousStateShift
                    );

                for (var h = 0; h < _currentLayerContainer.Configuration.SpatialDimension.Height; h++)
                {
                    for (var w = 0; w < _currentLayerContainer.Configuration.SpatialDimension.Width; w++)
                    {
                        var sum = 0f;

                        for (var hp = h * _currentLayerContainer.Configuration.InverseScaleFactor; hp < (h * _currentLayerContainer.Configuration.InverseScaleFactor) + _currentLayerContainer.Configuration.InverseScaleFactor; hp++)
                        {
                            for (var wp = w * _currentLayerContainer.Configuration.InverseScaleFactor; wp < (w * _currentLayerContainer.Configuration.InverseScaleFactor) + _currentLayerContainer.Configuration.InverseScaleFactor; wp++)
                            {
                                sum += previousState.GetValueFromCoordSafely(wp, hp);
                            }
                        }

                        sum /= _currentLayerContainer.Configuration.InverseScaleFactor * _currentLayerContainer.Configuration.InverseScaleFactor;

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