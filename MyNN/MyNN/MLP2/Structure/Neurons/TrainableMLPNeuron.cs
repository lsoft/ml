using System;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Neurons
{
    [Serializable]
    public abstract class TrainableMLPNeuron
    {
        public float[] Weights
        {
            get;
            protected set;
        }

        public IFunction ActivationFunction
        {
            get;
            protected set;
        }

        public abstract bool IsBiasNeuron
        {
            get;
        }
    }
}
