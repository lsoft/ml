using System;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;

namespace MyNN.MLP2.Structure.Neurons
{
    [Serializable]
    public class HiddeonOutputMLPNeuron : INeuron
    {
        public float[] Weights
        {
            get;
            private set;
        }

        public IFunction ActivationFunction
        {
            get;
            private set;
        }

        public bool IsBiasNeuron
        {
            get
            {
                return false;
            }
        }

        public INeuronConfiguration GetConfiguration()
        {
            return
                new NeuronConfiguration(
                    this.Weights.Length,
                    this.IsBiasNeuron);
        }

        public HiddeonOutputMLPNeuron(
            IFunction activationFunction,
            int weightCount,
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            this.ActivationFunction = activationFunction;

            //случайные веса
            this.Weights = new float[weightCount];
            for (var cc = 0; cc < weightCount; cc++)
            {
                this.Weights[cc] = randomizer.Next() * .2f - .1f;
            }
        }
    }
}
