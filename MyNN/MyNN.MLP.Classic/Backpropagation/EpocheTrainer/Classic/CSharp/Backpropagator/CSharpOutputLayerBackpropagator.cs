using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Backpropagator
{
    public class CSharpOutputLayerBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerContainer;
        private readonly ICSharpDesiredValuesContainer _desiredValuesContainer;
        private readonly ICSharpDeDyAggregator _deDyAggregator;

        private readonly float[] _nablaWeights;
        private readonly float[] _nablaBias;

        private readonly OutputLayerKernel _outputLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;

        public CSharpOutputLayerBackpropagator(
            ILearningAlgorithmConfig config,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpDesiredValuesContainer desiredValuesContainer,
            ICSharpDeDyAggregator deDyAggregator
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
            if (desiredValuesContainer == null)
            {
                throw new ArgumentNullException("desiredValuesContainer");
            }
            if (deDyAggregator == null)
            {
                throw new ArgumentNullException("deDyAggregator");
            }
            if (currentLayerContainer.Configuration.TotalNeuronCount != deDyAggregator.TotalNeuronCount)
            {
                throw new ArgumentException("Не совпадает число нейронов текущего слоя и число нейронов в dedy аггрегаторе");
            }

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _desiredValuesContainer = desiredValuesContainer;
            _deDyAggregator = deDyAggregator;

            _nablaWeights = new float[
                currentLayerContainer.Configuration.TotalNeuronCount * previousLayerContainer.Configuration.TotalNeuronCount
                ];
            _nablaBias = new float[currentLayerContainer.Configuration.TotalNeuronCount];

            _outputLayerKernel = new OutputLayerKernel(
                currentLayerContainer.Configuration,
                config
                );

            _updateWeightKernel = new UpdateWeightKernel();

        }

        public void Prepare()
        {
            this._deDyAggregator.ClearAndWrite();
        }

        public void Backpropagate(
            int dataCount, 
            float learningRate, 
            bool firstItemInBatch
            )
        {
            if (firstItemInBatch)
            {
                _outputLayerKernel.CalculateOverwrite(
                    _currentLayerContainer.NetMem,
                    _previousLayerContainer.StateMem,
                    _currentLayerContainer.StateMem,
                    this._deDyAggregator.DeDz,
                    _desiredValuesContainer.DesiredOutput,
                    _currentLayerContainer.WeightMem,
                    _nablaWeights,
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
                _outputLayerKernel.CalculateIncrement(
                    _currentLayerContainer.NetMem,
                    _previousLayerContainer.StateMem,
                    _currentLayerContainer.StateMem,
                    this._deDyAggregator.DeDz,
                    _desiredValuesContainer.DesiredOutput,
                    _currentLayerContainer.WeightMem,
                    _nablaWeights,
                    _previousLayerContainer.Configuration.TotalNeuronCount,
                    _currentLayerContainer.Configuration.TotalNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float)(dataCount),
                    _currentLayerContainer.BiasMem,
                    _nablaBias
                    );
            }

            this._deDyAggregator.Aggregate();
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