using System;
using System.Linq;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.OutputConsole;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP
{
    /// <summary>
    /// Вычисляет количество нулей в среднем по датасету на конечном слое нейросети
    /// </summary>
    public class SparseCalculator
    {
        private readonly Func<CLProvider, IPropagatorComponentConstructor> _pccFunc;
        private readonly IMLP _mlp;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="pccFunc">Создатель компонентов для форвардера</param>
        /// <param name="mlp">Сеть, через которую прогоняются данные</param>
        public SparseCalculator(
            Func<CLProvider, IPropagatorComponentConstructor> pccFunc,
            IMLP mlp)
        {
            if (pccFunc == null)
            {
                throw new ArgumentNullException("pccFunc");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            _pccFunc = pccFunc;
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
                var cc = _pccFunc(
                    clProvider
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                cc.CreateComponents(
                    _mlp,
                    out containers,
                    out propagators);

                var forward = new ForwardPropagation.ForwardPropagation(
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
