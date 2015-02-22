using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Backpropagator
{
    public class CSharpAvgPoolingFullConnectedBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly IAvgPoolingLayer _currentLayer;
        private readonly ICSharpLayerContainer _currentLayerContainer;
        private readonly ICSharpLayerContainer _nextLayerContainer;
        private readonly float[] _nextLayerDeDz;

        private readonly AvgPoolingFullConnectedKernel _kernel;

        public float[] DeDz
        {
            get;
            private set;
        }

        public CSharpAvgPoolingFullConnectedBackpropagator(
            IAvgPoolingLayer currentLayer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpLayerContainer nextLayerContainer,
            float[] nextLayerDeDz

            )
        {
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }
            if (nextLayerContainer == null)
            {
                throw new ArgumentNullException("nextLayerContainer");
            }
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }

            _currentLayer = currentLayer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerContainer = nextLayerContainer;
            _nextLayerDeDz = nextLayerDeDz;

            this.DeDz = new float[_currentLayer.GetConfiguration().TotalNeuronCount];

            _kernel = new AvgPoolingFullConnectedKernel(currentLayer);
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
