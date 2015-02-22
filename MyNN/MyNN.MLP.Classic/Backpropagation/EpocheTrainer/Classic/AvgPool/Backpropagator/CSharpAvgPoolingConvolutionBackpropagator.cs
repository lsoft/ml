using System;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Backpropagator
{
    public class CSharpAvgPoolingConvolutionBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly IAvgPoolingLayer _currentLayer;

        public float[] DeDz
        {
            get;
            private set;
        }

        public CSharpAvgPoolingConvolutionBackpropagator(
            IAvgPoolingLayer currentLayer
            )
        {
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }

            _currentLayer = currentLayer;

            this.DeDz = new float[_currentLayer.GetConfiguration().TotalNeuronCount];
        }


        public void Prepare()
        {
            //nothing to do
        }

        public void Backpropagate(int dataCount, float learningRate, bool firstItemInBatch)
        {
            for (var fmi = 0; fmi < _currentLayer.FeatureMapCount; fmi++)
            {
                var currentNetStateDeDzShift = fmi * _currentLayer.SpatialDimension.Multiplied;

                var dedz = new ReferencedSquareFloat(
                    _currentLayer.SpatialDimension,
                    this.DeDz,
                    currentNetStateDeDzShift
                    );

                _kernel.Calculate(
                    fmi * _currentLayer.SpatialDimension.Multiplied,
                    dedz,
                    _nextLayerDeDz,
                    _nextLayerContainer.WeightMem
                    );
            }
        }

        public void UpdateWeights()
        {
            //nothing to do
        }
    }
}