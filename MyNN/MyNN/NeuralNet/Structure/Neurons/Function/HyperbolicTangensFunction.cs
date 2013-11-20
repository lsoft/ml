using System;
using System.Globalization;

namespace MyNN.NeuralNet.Structure.Neurons.Function
{
    [Serializable]
    public class HyperbolicTangensFunction : IFunction
    {

        private readonly float _alpha = 0.0f;
        private readonly float _beta = 0.0f;

        public HyperbolicTangensFunction()
            : this(1.7159f, 0.6666f)
        {
        }

        public HyperbolicTangensFunction(float alpha, float beta)
        {
            _alpha = alpha;
            _beta = beta;
        }

        public float Compute(float x)
        {
            return 
                (float)(_alpha * Math.Tanh(_beta * x));
        }

        public float ComputeFirstDerivative(float t)
        {
            return (float)(_alpha * (
                _beta
                - Math.Pow(_beta, 3) * Math.Pow(t, 2)
                + 2.0 * Math.Pow(_beta, 5) * Math.Pow(t, 4) / 3.0
                - 17.0 * Math.Pow(_beta, 7) * Math.Pow(t, 6) / 45.0
                ));
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format("({0} * tanh({1} * {2}) )",
                      _alpha.ToString(CultureInfo.InvariantCulture),
                      _beta.ToString(CultureInfo.InvariantCulture),
                      varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            return
                string.Format(@"
({0} * (
    {1} 
    - pow({1}, 3) * pown({2}, 2)
    + 2.0f * pow({1}, 5) * pown({2}, 4) / 3.0f
    - 17.0f * pow({1}, 7) * pown({2}, 6) / 45.0f
    )
)",
                      _alpha.ToString(CultureInfo.InvariantCulture),
                      _beta.ToString(CultureInfo.InvariantCulture),
                      varName);

        }
    }

}
