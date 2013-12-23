using System;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Neurons
{
    [Serializable]
    public class InputMLPNeuron : 
        TrainableMLPNeuron
    {
        private readonly int _thisIndex;

        public InputMLPNeuron(
            IFunction activationFunction,
            int thisIndex)
        {
            this._thisIndex = thisIndex;
            this.ActivationFunction = activationFunction;

            //случайные веса
            this.Weights = new float[0];
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
