using System;

namespace MyNN.MLP2.Structure.Neurons.Function
{
    [Serializable]
    public class DRLUFunction : IFunction
    {

        public DRLUFunction()
        {
        }

        public string ShortName
        {
            get
            {
                return "IRLU";
            }
        }

        public float Compute(float x)
        {
            return
                Math.Max(0, x);
        }

        public float ComputeFirstDerivative(float computed)
        {
            //var sc = Math.Abs(computed);

            //if (Math.Abs(sc) < 1f)
            //{
            //    return
            //        1f;
            //}

            //return
            //    1f/sc;

            return 1f;
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
                //string.Format(
                //    "(fabs({0}) < 1.0 ? 1.0 : (1.0 / fabs({0})))",
                //    varName);
                "(1.0)";
        }
    }
}
