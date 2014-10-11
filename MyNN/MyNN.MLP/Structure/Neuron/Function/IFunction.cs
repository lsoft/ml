using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Structure.Neuron.Function
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

        /// <summary>
        /// Получить функцию активации как метод
        /// </summary>
        /// <param name="methodName">Название метода</param>
        /// <param name="vse">Тип векторизации, применяемый в кернеле</param>
        /// <returns>Текст метода</returns>
        string GetOpenCLActivationMethod(
            string methodName,
            VectorizationSizeEnum vse
            );

        /// <summary>
        /// Получить функцию производной как метод
        /// </summary>
        /// <param name="methodName">Название метода</param>
        /// <param name="vse">Тип векторизации, применяемый в кернеле</param>
        /// <returns>Текст метода</returns>
        string GetOpenCLDerivativeMethod(
            string methodName,
            VectorizationSizeEnum vse
            );
    }
}
