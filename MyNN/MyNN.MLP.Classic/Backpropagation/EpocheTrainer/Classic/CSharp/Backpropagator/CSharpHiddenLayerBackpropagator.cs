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
        private readonly bool _needToCalculateDeDy;

        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerContainer;
        private readonly ICSharpDeDyAggregator _nextLayerDeDyAggregator;
        private readonly ICSharpDeDyAggregator _currentLayerDeDyAggregator;

        private readonly float[] _nablaWeights;
        private readonly float[] _nablaBias;

        private readonly HiddenLayerKernel _hiddenLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;


        public CSharpHiddenLayerBackpropagator(
            ILearningAlgorithmConfig config,
            bool needToCalculateDeDy,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpDeDyAggregator nextLayerDeDyAggregator,
            ICSharpDeDyAggregator currentLayerDeDyAggregator
            )
        {
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

            _config = config;
            _needToCalculateDeDy = needToCalculateDeDy;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerDeDyAggregator = nextLayerDeDyAggregator;
            _currentLayerDeDyAggregator = currentLayerDeDyAggregator;

            _nablaWeights = new float[
                currentLayerContainer.Configuration.TotalNeuronCount * previousLayerContainer.Configuration.TotalNeuronCount
                ];
            _nablaBias = new float[currentLayerContainer.Configuration.TotalNeuronCount];

            _hiddenLayerKernel = new HiddenLayerKernel(
                currentLayerContainer.Configuration
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
                    _previousLayerContainer.Configuration.TotalNeuronCount,
                    _currentLayerContainer.Configuration.TotalNeuronCount,
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
                    _previousLayerContainer.Configuration.TotalNeuronCount,
                    _currentLayerContainer.Configuration.TotalNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float) (dataCount),
                    _currentLayerContainer.BiasMem,
                    _nablaBias
                    );
            }

            if (_needToCalculateDeDy)
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