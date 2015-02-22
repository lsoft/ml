using System;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron
{
    [Serializable]
    public class InputNeuron : INeuron
    {
        private readonly int _thisIndex;

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

        public InputNeuron(
            int thisIndex)
        {
            this._thisIndex = thisIndex;

            this.Bias = 0f;
            this.Weights = new float[0];
        }

        public INeuronConfiguration GetConfiguration()
        {
            return
                new NeuronConfiguration(
                    this.Weights.Length
                    );
        }

    }
}
