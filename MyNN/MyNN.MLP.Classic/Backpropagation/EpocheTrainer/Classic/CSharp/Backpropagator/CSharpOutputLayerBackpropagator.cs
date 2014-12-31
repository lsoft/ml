using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel;
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
        private readonly ILayer _outputLayer;
        private readonly ILayer _preOutputLayer;
        
        private readonly float[] _nablaWeights;
        private readonly float[] _nablaBias;

        private readonly OutputLayerKernel _outputLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;

        public float[] DeDz
        {
            get;
            private set;
        }


        public CSharpOutputLayerBackpropagator(
            IMLP mlp,
            ILearningAlgorithmConfig config,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpDesiredValuesContainer desiredValuesContainer
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
            if (desiredValuesContainer == null)
            {
                throw new ArgumentNullException("desiredValuesContainer");
            }

            _config = config;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _desiredValuesContainer = desiredValuesContainer;

            var layerIndex = mlp.Layers.Length - 1;

            _outputLayer = mlp.Layers[layerIndex];
            _preOutputLayer = mlp.Layers[layerIndex - 1];

            _nablaWeights = new float[
                _outputLayer.TotalNeuronCount * _preOutputLayer.TotalNeuronCount //_outputLayer.Neurons[0].Weights.Length
                ];
            _nablaBias = new float[_outputLayer.TotalNeuronCount];

            DeDz = new float[_outputLayer.TotalNeuronCount];

            _outputLayerKernel = new OutputLayerKernel(
                _outputLayer,
                config
                );

            _updateWeightKernel = new UpdateWeightKernel();

        }

        public void Prepare()
        {
            //nothing to do
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
                    this.DeDz,
                    _desiredValuesContainer.DesiredOutput,
                    _currentLayerContainer.WeightMem,
                    _nablaWeights,
                    _preOutputLayer.TotalNeuronCount,
                    _outputLayer.TotalNeuronCount,
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
                    this.DeDz,
                    _desiredValuesContainer.DesiredOutput,
                    _currentLayerContainer.WeightMem,
                    _nablaWeights,
                    _preOutputLayer.TotalNeuronCount,
                    _outputLayer.TotalNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float)(dataCount),
                    _currentLayerContainer.BiasMem,
                    _nablaBias
                    );
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