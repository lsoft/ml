using System;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.DBNInfo.WeightLoader
{
    public class RBMAutoencoderWeightLoader : IWeightLoader
    {
        private readonly string _pathToWeightsFile;
        private readonly ISerializationHelper _serializationHelper;

        public RBMAutoencoderWeightLoader(
            string pathToWeightsFile,
            ISerializationHelper serializationHelper
            )
        {
            if (pathToWeightsFile == null)
            {
                throw new ArgumentNullException("pathToWeightsFile");
            }
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }
            if (string.IsNullOrEmpty(pathToWeightsFile))
            {
                throw new ArgumentException("string.IsNullOrEmpty(pathToWeightsFile)");
            }

            _pathToWeightsFile = pathToWeightsFile;
            _serializationHelper = serializationHelper;
        }

        public void LoadWeights(ILayer layer)
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            ConsoleAmbientContext.Console.WriteWarning(
                "Не проверено! Проверить! Причем не используются HiddenBias - выяснить нормально ли это."
                );

            var sc = _serializationHelper.LoadFromFile<SaveableContainer>(_pathToWeightsFile);

            for (var neuronIndex = 0; neuronIndex < layer.TotalNeuronCount; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                if (neuron is HiddeonOutputMLPNeuron)
                {
                    for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        var weight = sc.Weights[weightIndex * layer.TotalNeuronCount + neuronIndex];
                        neuron.Weights[weightIndex] = weight;
                    }

                    neuron.Bias = sc.VisibleBiases[neuronIndex];
                }
            }
        }
    }
}