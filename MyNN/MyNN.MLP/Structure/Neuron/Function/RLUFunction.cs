using System;
using MyNN.Common.OpenCLHelper;

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

        public string GetOpenCLActivationMethod(
            string methodName,
            VectorizationSizeEnum vse
            )
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv zero = 0.0;

    floatv result = max(zero, incoming);

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
inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv one = 1.0;
    const floatv zero = 0.0;
            
    floatv result = (incoming > zero) ? one : zero;
            
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
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }
    }
}
