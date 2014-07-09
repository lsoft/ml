using System;

namespace MyNN.MLP2.Structure.Layer.Factory.WeightLoader
{
    public class RBMWeightLoader : IWeightLoader
    {
        private readonly string _pathToWeightsFile;

        public RBMWeightLoader(
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

            //загружаем веса
            var weightFile = SerializationHelper.LoadFromFile<float[]>(_pathToWeightsFile);
            var lineIndex = 0;
            for (var neuronIndex = 0; neuronIndex < layer.Neurons.Length; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++, lineIndex++)
                {
                    if (!neuron.IsBiasNeuron)
                    {
                        var weight = weightFile[lineIndex];
                        neuron.Weights[weightIndex] = weight;
                    }
                }
            }
        }
    }
}