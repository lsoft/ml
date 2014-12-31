using System;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron
{
    [Serializable]
    public class HiddeonOutputMLPNeuron : INeuron
    {
        public float[] Weights
        {
            get;
            private set;
        }

        public float Bias
        {
            get;
            set;
        }

        public INeuronConfiguration GetConfiguration()
        {
            return
                new NeuronConfiguration(
                    this.Weights.Length
                    );
        }

        public HiddeonOutputMLPNeuron(
            int weightCount,
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            //случайные веса
            this.Weights = new float[weightCount];
            for (var cc = 0; cc < weightCount; cc++)
            {
                this.Weights[cc] = randomizer.Next() * .2f - .1f;
            }

            this.Bias = randomizer.Next() * .2f - .1f;
        }
    }
}
