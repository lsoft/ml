using System;
using MyNN.MLP2.Structure.Neurons;

namespace MyNN.MLP2.Structure.Layer.Factory.WeightLoader
{
    public class RBMAutoencoderWeightLoader : IWeightLoader
    {
        private readonly string _pathToWeightsFile;

        public RBMAutoencoderWeightLoader(
            string pathToWeightsFile)
        {
            if (pathToWeightsFile == null)
            {
                throw new ArgumentNullException("pathToWeightsFile");
            }
            if (string.IsNullOrEmpty(pathToWeightsFile))
            {
                throw new ArgumentException("string.IsNullOrEmpty(pathToWeightsFile)");
            }

            _pathToWeightsFile = pathToWeightsFile;
        }

        public void LoadWeights(ILayer layer)
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            var weightFile = SerializationHelper.LoadFromFile<float[]>(_pathToWeightsFile);

            for (var neuronIndex = 0; neuronIndex < layer.Neurons.Length; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                if (neuron is HiddeonOutputMLPNeuron)
                {
                    for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        var weight = weightFile[weightIndex * (layer.NonBiasNeuronCount + 1) + neuronIndex];
                        neuron.Weights[weightIndex] = weight;
                    }
                }
            }
        }
    }
}