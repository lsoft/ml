using System;
using System.Collections.Generic;
using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet.Computers
{
    public class DefaultComputer : IMultilayerComputer
    {
        private readonly MultiLayerNeuralNetwork _net;

        public DefaultComputer(
            MultiLayerNeuralNetwork net)
        {
            _net = net;
        }

        public List<float[]> ComputeOutput(List<float[]> inputVectors)
        {
            var result = new List<float[]>();

            foreach (var inputVector in inputVectors)
            {
                var r = this.ComputeOutput(inputVector);
                result.Add(r);
            }

            return result;
        }

        public float[] ComputeOutput(
            float[] inputVector)
        {
            var calcResult = new float[inputVector.Length];
            Array.Copy(inputVector, calcResult, inputVector.Length);

            for (var cc = 0; cc < _net.Layers.Length; cc++)
            {
                var tmp = _net.Layers[cc].Compute(calcResult);

                calcResult = new float[tmp.Length];
                Array.Copy(tmp, calcResult, tmp.Length);
            }

            return calcResult;
        }

        public void ExecuteComputation()
        {
            throw new InvalidOperationException("Неприменимо к стандартному просчетчику");
        }

    }
}
