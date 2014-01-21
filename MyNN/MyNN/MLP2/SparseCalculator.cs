using System;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using OpenCL.Net.OpenCL;

namespace MyNN.MLP2
{
    /// <summary>
    /// Вычисляет количество нулей в среднем по датасету на конечном слое нейросети
    /// </summary>
    public class SparseCalculator
    {
        private readonly MLP _mlp;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="mlp">Сеть, через которую прогоняются данные</param>
        public SparseCalculator(
            MLP mlp)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            _mlp = mlp;
        }

        /// <summary>
        /// Просчет доли нулей
        /// </summary>
        /// <param name="testDataset">Проверочные данные (обычно валидационные)</param>
        /// <param name="sparsePart">Доля нулей по всем нейронам выходного слоя и по всему датасету (0-1)</param>
        /// <param name="avgNonZeroCountPerItem">Среднее количество не нулевых флоатов на один тестовый пример</param>
        /// <param name="avgValueOfNonZero">Среднее значение не нулевых флоатов</param>
        public void Calculate(
            DataSet testDataset,
            out float sparsePart,
            out float avgNonZeroCountPerItem,
            out float avgValueOfNonZero)
        {
            if (testDataset == null)
            {
                throw new ArgumentNullException("testDataset");
            }

            ConsoleAmbientContext.Console.WriteLine(_mlp.DumpLayerInformation());

            using (var clProvider = new CLProvider())
            {
                var forward = new OpenCLForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    _mlp,
                    clProvider);

                var sparseddata = forward.ComputeOutput(testDataset);

                var totalZero = sparseddata.Sum(j => j.Count(k => Math.Abs(k) < float.Epsilon));
                sparsePart = totalZero / (float)sparseddata.Count / (float)sparseddata[0].State.Length;

                avgNonZeroCountPerItem = (float)sparseddata.Average(j => j.State.Count(k => Math.Abs(k) >= float.Epsilon));
                avgValueOfNonZero = sparseddata.Average(j => j.State.Where(k => Math.Abs(k) >= float.Epsilon).Average());
            }
        }
    }
}
