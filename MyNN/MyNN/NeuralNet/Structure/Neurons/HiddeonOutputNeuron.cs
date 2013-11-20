using System;
using System.Linq;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure.Neurons
{
    [Serializable]
    public class HiddeonOutputNeuron : 
        TrainableNeuron
    {
        public HiddeonOutputNeuron(
            IFunction activationFunction,
            int weightCount,
            ref int seed)
        {
            this.ActivationFunction = activationFunction;
            this.Dedz = 0.0f;

            var rnd = new Random(seed++);

            //случайные веса
            this.Weights = new float[weightCount];
            for (var cc = 0; cc < weightCount; cc++)
            {
                this.Weights[cc] = (float)(rnd.NextDouble() * .2 - .1);
            }
        }

        /// <summary>
        /// Compute NET of the neuron by input vector
        /// </summary>
        /// <param name="inputVector">Input vector</param>
        /// <returns>Compute NET of neuron</returns>
        private float ComputeNET(float[] inputVector)
        {
            var sum = 0.0f;

            for (var cc = 0; cc < inputVector.Length; ++cc)
            {
                sum += Weights[cc] * inputVector[cc];
            }

            this.LastNET = sum;

            return this.LastNET;
        }

        public override float Activate(float[] inputVector)
        {
            var sum = this.ComputeNET(inputVector);
            this.LastState = this.ActivationFunction.Compute(sum);

            return this.LastState;
        }

        public override string ToString()
        {
            var part1 =
                string.Format(
                    "LastNET = {0}, LastState = {1}, Dedz = {2}",
                    this.LastNET,
                    this.LastState,
                    this.Dedz);

            var part2 = string.Join(",", this.Weights.ToArray());

            return
                string.Format("N {0}, [{1}]", part1, part2);
        }

    }
}
