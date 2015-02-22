using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Backpropagator
{
    public class CSharpHiddenLayerBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly int _layerIndex;
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;

        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerContainer;
        private readonly ICSharpDeDyAggregator _nextLayerDeDyAggregator;
        private readonly ICSharpDeDyAggregator _currentLayerDeDyAggregator;

        private readonly float[] _nablaWeights;
        private readonly float[] _nablaBias;

        private readonly HiddenLayerKernel _hiddenLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;


        public CSharpHiddenLayerBackpropagator(
            IMLP mlp,
            ILearningAlgorithmConfig config,
            int layerIndex,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpLayerContainer nextLayerContainer,
            ICSharpDeDyAggregator nextLayerDeDyAggregator,
            ICSharpDeDyAggregator currentLayerDeDyAggregator
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (previousLayerContainer == null)
            {
                throw new ArgumentNullException("previousLayerContainer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }
            if (nextLayerContainer == null)
            {
                throw new ArgumentNullException("nextLayerContainer");
            }
            if (nextLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("nextLayerDeDyAggregator");
            }
            if (currentLayerDeDyAggregator == null)
            {
                throw new ArgumentNullException("currentLayerDeDyAggregator");
            }

            var previousLayer = mlp.Layers[layerIndex - 1];
            var currentLayer = mlp.Layers[layerIndex];

            _config = config;
            _layerIndex = layerIndex;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;

            _nablaWeights = new float[
                currentLayer.TotalNeuronCount*_previousLayer.TotalNeuronCount
                ];
            _nablaBias = new float[currentLayer.TotalNeuronCount];

            _hiddenLayerKernel = new HiddenLayerKernel(
                currentLayer,
                config
                );

            _updateWeightKernel = new UpdateWeightKernel();
        }

        public void Prepare()
        {
            this._currentLayerDeDyAggregator.ClearAndWrite();
        }

        public void Backpropagate(
            int dataCount,
            float learningRate,
            bool firstItemInBatch
            )
        {
            if (firstItemInBatch)
            {
                _hiddenLayerKernel.CalculateOverwrite(
                    _currentLayerContainer.NetMem,
                    _previousLayerContainer.StateMem,
                    this._currentLayerDeDyAggregator.DeDz,
                    _currentLayerContainer.WeightMem,
                    _nablaWeights,
                    _nextLayerDeDyAggregator.DeDy,
                    _previousLayer.TotalNeuronCount,
                    _currentLayer.TotalNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float)(dataCount),
                    _currentLayerContainer.BiasMem,
                    _nablaBias
                    );
            }
            else
            {
                _hiddenLayerKernel.CalculateIncrement(
                    _currentLayerContainer.NetMem,
                    _previousLayerContainer.StateMem,
                    this._currentLayerDeDyAggregator.DeDz,
                    _currentLayerContainer.WeightMem,
                    _nablaWeights,
                    _nextLayerDeDyAggregator.DeDy,
                    _previousLayer.TotalNeuronCount,
                    _currentLayer.TotalNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float) (dataCount),
                    _currentLayerContainer.BiasMem,
                    _nablaBias
                    );
            }

            if (_layerIndex > 1)
            {
                this._currentLayerDeDyAggregator.Aggregate();
            }
        }

        public void UpdateWeights()
        {
            var weightMem = _currentLayerContainer.WeightMem;
            var nablaWeights = _nablaWeights;

            var biasMem = _currentLayerContainer.BiasMem;
            var nablaBias = _nablaBias;

            _updateWeightKernel.UpdateWeigths(
                weightMem,
                nablaWeights,
                (float)(_config.BatchSize),
                biasMem,
                nablaBias
                );
        }
    }
}