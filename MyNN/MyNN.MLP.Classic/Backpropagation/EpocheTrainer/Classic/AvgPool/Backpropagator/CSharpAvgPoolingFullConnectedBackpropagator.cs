using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Backpropagator
{
    public class CSharpAvgPoolingFullConnectedBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ICSharpAvgPoolingLayerContainer _currentLayerContainer;
        private readonly ICSharpDeDyAggregator _nextLayerDeDyAggregator;
        private readonly ICSharpDeDyAggregator _currentLayerDeDyAggregator;

        private readonly AvgPoolingFullConnectedKernel _kernel;

        public CSharpAvgPoolingFullConnectedBackpropagator(
            ICSharpAvgPoolingLayerContainer currentLayerContainer,
            ICSharpDeDyAggregator nextLayerDeDyAggregator,
            ICSharpDeDyAggregator currentLayerDeDyAggregator
            )
        {
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }
            if (nextLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("nextLayerDeDyAggregator");
            }
            if (currentLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("currentLayerDeDyAggregator");
            }
            if (currentLayerContainer.Configuration.TotalNeuronCount != currentLayerDeDyAggregator.TotalNeuronCount)
            {
                throw new ArgumentException("Не совпадает число нейронов текущего слоя и число нейронов в dedy аггрегаторе");
            }

            _currentLayerContainer = currentLayerContainer;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;

            _kernel = new AvgPoolingFullConnectedKernel(
                _currentLayerContainer.Configuration
                );
        }

        public void Prepare()
        {
            //nothing to do
        }

        public void Backpropagate(int dataCount, float learningRate, bool firstItemInBatch)
        {
            for (var fmi = 0; fmi < _currentLayerContainer.Configuration.FeatureMapCount; fmi++)
            {
                var currentNetStateDeDzShift = fmi * _currentLayerContainer.Configuration.SpatialDimension.Multiplied;

                var dedz = new ReferencedSquareFloat(
                    _currentLayerContainer.Configuration.SpatialDimension,
                    _currentLayerDeDyAggregator.DeDz,
                    currentNetStateDeDzShift
                    );

                _kernel.Calculate(
                    fmi * _currentLayerContainer.Configuration.SpatialDimension.Multiplied,
                    dedz,
                    _nextLayerDeDyAggregator.DeDy
                    );
            }

            _currentLayerDeDyAggregator.Aggregate();
        }

        public void UpdateWeights()
        {
            //nothing to do
        }

    }
}
