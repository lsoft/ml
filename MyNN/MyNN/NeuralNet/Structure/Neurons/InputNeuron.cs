using System;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure.Neurons
{
    [Serializable]
    public class InputNeuron : 
        TrainableNeuron
    {
        private readonly int _thisIndex;

        public InputNeuron(
            IFunction activationFunction,
            int thisIndex)
        {
            _thisIndex = thisIndex;
            this.ActivationFunction = activationFunction;
            this.LastNET = 0.0f;
            this.Dedz = 0.0f;

            //случайные веса
            this.Weights = new float[0];
        }

        public override float Activate(float[] inputVector)
        {
            this.LastState = inputVector[_thisIndex];

            return this.LastState;
        }

        public override string ToString()
        {
            return "IN LastState = " + this.LastState;
        }
    }
}
