using System;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DBNInfo.WeightLoader
{
    public class RBMWeightLoader : IWeightLoader
    {
        private readonly string _pathToWeightsFile;
        private readonly ISerializationHelper _serializationHelper;

        public RBMWeightLoader(
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

            //загружаем веса
            var sc = _serializationHelper.LoadFromFile<SaveableContainer>(_pathToWeightsFile);
            var lineIndex = 0;
            for (var neuronIndex = 0; neuronIndex < layer.TotalNeuronCount; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++, lineIndex++)
                {
                    var weight = sc.Weights[lineIndex];
                    neuron.Weights[weightIndex] = weight;
                }

                neuron.Bias = sc.VisibleBiases[neuronIndex];
            }
        }
    }
}