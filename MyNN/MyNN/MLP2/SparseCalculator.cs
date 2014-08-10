using System;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2
{
    /// <summary>
    /// Вычисляет количество нулей в среднем по датасету на конечном слое нейросети
    /// </summary>
    public class SparseCalculator
    {
        private readonly IMLP _mlp;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="mlp">Сеть, через которую прогоняются данные</param>
        public SparseCalculator(
            IMLP mlp)
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
            IDataSet testDataset,
            out float sparsePart,
            out float avgNonZeroCountPerItem,
            out float avgValueOfNonZero)
        {
            if (testDataset == null)
            {
                throw new ArgumentNullException("testDataset");
            }

            ConsoleAmbientContext.Console.WriteLine(_mlp.GetLayerInformation());

            using (var clProvider = new CLProvider())
            {
                var cc = new CPUPropagatorComponentConstructor(
                    clProvider,
                    VectorizationSizeEnum.VectorizationMode16
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                cc.CreateComponents(
                    _mlp,
                    out containers,
                    out propagators);

                var forward = new ForwardPropagation2(
                    containers,
                    propagators,
                    _mlp
                    );

                var sparseddata = forward.ComputeOutput(testDataset);

                var totalZero = sparseddata.Sum(j => j.Count(k => Math.Abs(k) < float.Epsilon));
                sparsePart = totalZero / (float)sparseddata.Count / (float)sparseddata[0].NState.Length;

                avgNonZeroCountPerItem = (float)sparseddata.Average(j => j.NState.Count(k => Math.Abs(k) >= float.Epsilon));
                avgValueOfNonZero = sparseddata.Average(j => j.NState.Where(k => Math.Abs(k) >= float.Epsilon).Average());
            }
        }
    }
}
