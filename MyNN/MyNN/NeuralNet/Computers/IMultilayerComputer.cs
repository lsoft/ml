using System.Collections.Generic;
using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet.Computers
{
    public interface IMultilayerComputer
    {
        List<float[]> ComputeOutput(
            List<float[]> inputVector);

        float[] ComputeOutput(
            float[] inputVector);

        void ExecuteComputation();
    }
}