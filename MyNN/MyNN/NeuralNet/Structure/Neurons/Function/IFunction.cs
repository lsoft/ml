namespace MyNN.NeuralNet.Structure.Neurons.Function
{
    public interface IFunction
    {
        float Compute(float x);
        float ComputeFirstDerivative(float x);

        string GetOpenCLFirstDerivative(string varName);
        string GetOpenCLActivationFunction(string varName);
    }
}
