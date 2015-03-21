using System;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Backpropagator
{
    public class CSharpAvgPoolingConvolutionBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly IAvgPoolingLayerConfiguration _currentLayerConfiguration;
        private readonly ICSharpDeDyAggregator _nextLayerDeDyAggregator;
        private readonly ICSharpDeDyAggregator _currentLayerDeDyAggregator;

        private readonly AvgPoolingConvolutionKernel _kernel;

        public CSharpAvgPoolingConvolutionBackpropagator(
            IAvgPoolingLayerConfiguration currentLayerConfiguration,
            ICSharpDeDyAggregator nextLayerDeDyAggregator,
            ICSharpDeDyAggregator currentLayerDeDyAggregator
            )
        {
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }
            if (nextLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("nextLayerDeDyAggregator");
            }
            if (currentLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("currentLayerDeDyAggregator");
            }
            if (currentLayerConfiguration.TotalNeuronCount != currentLayerDeDyAggregator.TotalNeuronCount)
            {
                throw new ArgumentException("Не совпадает число нейронов текущего слоя и число нейронов в dedy аггрегаторе");
            }

            _currentLayerConfiguration = currentLayerConfiguration;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;

            _kernel = new AvgPoolingConvolutionKernel(
                currentLayerConfiguration
                );
        }


        public void Prepare()
        {
            //nothing to do
        }

        public void Backpropagate(int dataCount, float learningRate, bool firstItemInBatch)
        {
            for (var fmi = 0; fmi < _currentLayerConfiguration.FeatureMapCount; fmi++)
            {
                var currentDeDzShift = fmi * _currentLayerConfiguration.SpatialDimension.Multiplied;

                var currentLayerDeDz = new ReferencedSquareFloat(
                    _currentLayerConfiguration.SpatialDimension,
                    this._currentLayerDeDyAggregator.DeDz,
                    currentDeDzShift
                    );

                var nextDeDyShift = fmi * _currentLayerConfiguration.SpatialDimension.Multiplied;

                var nextDeDy = new ReferencedSquareFloat(
                    _currentLayerConfiguration.SpatialDimension,
                    _nextLayerDeDyAggregator.DeDy,
                    nextDeDyShift
                    );

                _kernel.Calculate(
                    currentLayerDeDz,
                    nextDeDy
                    );
            }
        }

        public void UpdateWeights()
        {
            //nothing to do
        }
    }
}