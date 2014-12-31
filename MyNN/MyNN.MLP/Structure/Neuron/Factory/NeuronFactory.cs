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

        public INeuron CreateInputNeuron(int thisIndex)
        {
            return
                new InputNeuron(thisIndex);
        }

        public INeuron CreateTrainableNeuron(
            int weightCount
            )
        {
            return
                new HiddeonOutputMLPNeuron(
                    weightCount,
                    _randomizer);
        }
    }
}