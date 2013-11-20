using System;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure.Neurons
{
    [Serializable]
    public class BiasNeuron : 
        TrainableNeuron
    {
        public BiasNeuron(IFunction activationFunction)
        {
            this.ActivationFunction = activationFunction;
            this.LastNET = 0.0f;
            this.LastState = 1.0f;
            this.Dedz = 0.0f;

            //случайные веса
            this.Weights = new float[0];
        }

        public override float Activate(float[] inputVector)
        {
            this.LastNET = 0.0f;
            this.LastState = 1.0f;
            this.Dedz = 0.0f;

            return this.LastState;
        }

        public override string ToString()
        {
            return "BN";
        }

    }
}
