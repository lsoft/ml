using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.NextLayerAggregator;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Backpropagator
{
    public class CSharpHiddenLayerBackpropagator : ICSharpLayerBackpropagator
    {
        private readonly ILearningAlgorithmConfig _config;
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;
        private readonly ILayer _nextLayer;

        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerContainer;
        private readonly ICSharpLayerContainer _nextLayerContainer;
        private readonly float[] _nextLayerDeDz;

        private readonly float[] _nablaWeights;
        private readonly float[] _currentDeDz;
        private readonly HiddenLayerKernel _hiddenLayerKernel;
        private readonly UpdateWeightKernel _updateWeightKernel;


        public float[] DeDz
        {
            get
            {
                return
                    _currentDeDz;
            }
        }

        public CSharpHiddenLayerBackpropagator(
            IMLP mlp,
            ILearningAlgorithmConfig config,
            int layerIndex,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer,
            ICSharpLayerContainer nextLayerContainer,
            float[] nextLayerDeDz
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
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }

            var previousLayer = mlp.Layers[layerIndex - 1];
            var currentLayer = mlp.Layers[layerIndex];
            var nextLayer = mlp.Layers[layerIndex + 1];

            _config = config;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _nextLayer = nextLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
            _nextLayerContainer = nextLayerContainer;
            _nextLayerDeDz = nextLayerDeDz;

            _nablaWeights = new float[
                currentLayer.NonBiasNeuronCount*currentLayer.Neurons[0].Weights.Length
                ];

            _currentDeDz = new float[currentLayer.NonBiasNeuronCount];

            _hiddenLayerKernel = new HiddenLayerKernel(
                currentLayer,
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
                _hiddenLayerKernel.CalculateOverwrite(
                    _currentLayerContainer.NetMem,
                    _previousLayerContainer.StateMem,
                    _currentLayerContainer.StateMem,
                    _currentDeDz,
                    _nextLayerDeDz,
                    _currentLayerContainer.WeightMem,
                    _nextLayerContainer.WeightMem,
                    _nablaWeights,
                    _previousLayer.Neurons.Length,
                    _currentLayer.NonBiasNeuronCount,
                    _nextLayer.NonBiasNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float)(dataCount)
                    );
            }
            else
            {
                _hiddenLayerKernel.CalculateIncrement(
                    _currentLayerContainer.NetMem,
                    _previousLayerContainer.StateMem,
                    _currentLayerContainer.StateMem,
                    _currentDeDz,
                    _nextLayerDeDz,
                    _currentLayerContainer.WeightMem,
                    _nextLayerContainer.WeightMem,
                    _nablaWeights,
                    _previousLayer.Neurons.Length,
                    _currentLayer.NonBiasNeuronCount,
                    _nextLayer.NonBiasNeuronCount,
                    learningRate,
                    _config.RegularizationFactor,
                    (float) (dataCount)
                    );
            }
        }

        public void UpdateWeights()
        {
            var weightMem = _currentLayerContainer.WeightMem;
            var nablaMem = _nablaWeights;

            _updateWeightKernel.UpdateWeigths(
                weightMem,
                nablaMem,
                (float)(_config.BatchSize)
                );
        }

    }
}