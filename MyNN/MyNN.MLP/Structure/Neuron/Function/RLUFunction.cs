using System;

namespace MyNN.MLP.Structure.Neuron.Function
{
    [Serializable]
    public class RLUFunction : IFunction
    {

        public RLUFunction()
        {
        }

        public string ShortName
        {
            get
            {
                return "RLU";
            }
        }

        public float Compute(float x)
        {
            return
                Math.Max(0, x);
        }

        public float ComputeFirstDerivative(float computed)
        {
            return
                computed > 0.0f ? 1f : 0f;
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format(
                    "max((float)(0.0), {0})",
                    varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            return
                string.Format(
                    "({0} > 0.0 ? 1.0 : 0.0)",
                    varName);
        }
    }
}
