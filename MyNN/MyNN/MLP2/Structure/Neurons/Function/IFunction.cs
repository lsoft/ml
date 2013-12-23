namespace MyNN.MLP2.Structure.Neurons.Function
{
    public interface IFunction
    {
        string ShortName
        {
            get;
        }

        float Compute(float x);
        float ComputeFirstDerivative(float x);

        string GetOpenCLFirstDerivative(string varName);
        string GetOpenCLActivationFunction(string varName);
    }
}
