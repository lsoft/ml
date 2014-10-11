using System;
using System.Globalization;
using System.IO;
using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Structure.Neuron.Function
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

        public string GetOpenCLFirstDerivative(string varName)
        {
            var computed = this.GetOpenCLActivationFunction(varName);

            return
                string.Format(
                    "({0} * {1} * (1.0 - {1}))",
                    _alpha.ToString(CultureInfo.InvariantCulture),
                    computed);
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format(
                    "((float)(1.0) / ((float)(1.0) + exp((float)(-{0}) * {1})))",
                    _alpha.ToString(CultureInfo.InvariantCulture),
                    varName);
        }

        public string GetOpenCLActivationMethod(
            string methodName,
            VectorizationSizeEnum vse)
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv one = 1.0;
    const floatv minusAlpha = -{ALPHA};

    floatv result = one / (one + exp(minusAlpha * incoming));

    return result;
}
";

            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);

            var result = methodBody;

            result = result.Replace(
                "floatv",
                string.Format(
                    "float{0}",
                    vsize));

            result = result.Replace(
                "{ALPHA}",
                _alpha.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{METHOD_NAME}", 
                methodName
                );

            return result;
        }

        public string GetOpenCLDerivativeMethod(
            string methodName, 
            VectorizationSizeEnum vse
            )
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
{ACTIVATION_METHOD}

inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv one = 1.0;
    const floatv alpha = {ALPHA};

    floatv activation = {ACTIVATION_METHOD_NAME}(incoming);

    floatv result = alpha * activation * (one - activation);

    return result;
}
";


            var activationKernelName =
                string.Format(
                    "ActivationKernel{0}",
                    Guid.NewGuid());

            activationKernelName = activationKernelName.Replace("-", "");

            var activationKernel = this.GetOpenCLActivationMethod(
                activationKernelName,
                vse);


            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);

            var result = methodBody;

            result = result.Replace(
                "{ACTIVATION_METHOD}",
                activationKernel
                );

            result = result.Replace(
                "{ACTIVATION_METHOD_NAME}",
                activationKernelName
                );

            result = result.Replace(
                "floatv",
                string.Format(
                    "float{0}",
                    vsize));

            result = result.Replace(
                "{ALPHA}",
                _alpha.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }
    }
}
