using System;
using System.Globalization;

namespace MyNN.MLP.Structure.Neuron.Function
{
    [Serializable]
    public class HyperbolicTangensFunction : IFunction
    {

        private readonly float _alpha = 0.0f;
        private readonly float _beta = 0.0f;

        public string ShortName
        {
            get
            {
                return "HTan";
            }
        }

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

        public float ComputeFirstDerivative(float x)
        {
            var computed = this.Compute(x);

            var scaled = computed/_alpha;

            return
                _alpha * _beta * (1f - scaled * scaled);
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format("((float)({0}) * tanh((float)({1}) * {2}) )",
                      _alpha.ToString(CultureInfo.InvariantCulture),
                      _beta.ToString(CultureInfo.InvariantCulture),
                      varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            var computed = this.GetOpenCLActivationFunction(varName);

            return
                string.Format(@"
({0} * {1} * (((float)1) - ({2} / {0}) * ({2} / {0}))
)",
                      _alpha.ToString(CultureInfo.InvariantCulture),
                      _beta.ToString(CultureInfo.InvariantCulture),
                      computed);

        }
    }

}
