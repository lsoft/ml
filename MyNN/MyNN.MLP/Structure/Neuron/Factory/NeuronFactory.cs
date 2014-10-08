using System;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron.Factory
{
    [Serializable]
    public class NeuronFactory : INeuronFactory
    {
        private readonly IRandomizer _randomizer;

        public NeuronFactory(
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
        }

        public INeuron CreateBiasNeuron()
        {
            return 
                new BiasNeuron();
        }

        public INeuron CreateInputNeuron(int thisIndex)
        {
            return
                new InputNeuron(thisIndex);
        }

        public INeuron CreateTrainableNeuron(
            IFunction activationFunction,
            int weightCount)
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            return
                new HiddeonOutputMLPNeuron(
                    activationFunction,
                    weightCount,
                    _randomizer);
        }
    }
}