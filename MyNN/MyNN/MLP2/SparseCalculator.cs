﻿using System;
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
        /// <param name="dataset">Проверочные данные (обычно валидационные)</param>
        /// <returns>Доля нулей по всем нейронам выходного слоя и по всему датасету (0-1)</returns>
        public float Calculate(
            DataSet dataset)
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }

            ConsoleAmbientContext.Console.WriteLine(_mlp.DumpLayerInformation());

            var sparsePart = 0f;
            using (var clProvider = new CLProvider())
            {
                var forward = new OpenCLForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    _mlp,
                    clProvider);

                var sparseddata = forward.ComputeOutput(dataset);
                var totalZero = sparseddata.Sum(j => j.Count(k => Math.Abs(k) < float.Epsilon));
                
                sparsePart = totalZero / (float)sparseddata.Count / (float)sparseddata[0].State.Length;
            }

            return sparsePart;
        }
    }
}
