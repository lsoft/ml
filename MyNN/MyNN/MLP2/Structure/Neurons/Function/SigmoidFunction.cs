using System;
using System.Globalization;

namespace MyNN.MLP2.Structure.Neurons.Function
{
    [Serializable]
    public class SigmoidFunction : IFunction
    {

        private readonly float _alpha = 1.0f;

        public string ShortName
        {
            get
            {
                return "Sigm";
            }
        }


        public SigmoidFunction(float alpha)
        {
            _alpha = alpha;
        }

        public float Compute(float x)
        {
            var r = (float)(1.0 / (1.0 + Math.Exp(-1.0 * _alpha * x)));
            return r;
        }

        public float ComputeFirstDerivative(float x)
        {
            var computed = this.Compute(x);

            return 
                (float)(_alpha * computed * (1.0 - computed));
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format(
                    //"(1.0 / (1.0 + exp(-1.0 * {0} * {1})))",
                    "((float)(1.0) / ((float)(1.0) + exp((float)(-{0}) * {1})))",
                    _alpha.ToString(CultureInfo.InvariantCulture),
                    varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            var computed = this.GetOpenCLActivationFunction(varName);

            return
                string.Format(
                    "({0} * {1} * (1.0 - {1}))", 
                    _alpha.ToString(CultureInfo.InvariantCulture),
                    computed);
        }
    }
}
