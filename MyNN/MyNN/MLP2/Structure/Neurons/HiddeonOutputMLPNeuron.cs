using System;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;

namespace MyNN.MLP2.Structure.Neurons
{
    [Serializable]
    public class HiddeonOutputMLPNeuron : 
        TrainableMLPNeuron
    {
        private readonly IRandomizer _randomizer;

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
            _randomizer = randomizer;

            //случайные веса
            this.Weights = new float[weightCount];
            for (var cc = 0; cc < weightCount; cc++)
            {
                this.Weights[cc] = _randomizer.Next() * .2f - .1f;
            }
        }

        public override bool IsBiasNeuron
        {
            get
            {
                return false;
            }
        }
    }
}
