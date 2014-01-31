using System;
using System.Collections.Generic;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.Boosting.SAMMEBoosting.EpocheTrainers.Classifiers
{
    internal class MLPClassifier : IEpocheClassifier
    {
        private readonly MLP _network;

        public MLPClassifier(MLP network)
        {
            if (network == null)
            {
                throw new ArgumentNullException("network");
            }

            _network = network;
        }

        public int Compute(double[] input)
        {

            ILayerState netResult = null;
            using (var clProvider = new CLProvider())
            {
                var forward = new CPUForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    _network,
                    clProvider);

                var inputf = input.ToList().ConvertAll(j => (float)j).ToArray();

                var ds = new DataSet(
                    new List<DataItem>
                    {
                        new DataItem(
                            inputf,
                            new float[]
                            {
                                0f
                            })
                    });

                var netResults = forward.ComputeOutput(ds);
                netResult = netResults[0];
            }

            var result = -1;

            //берем максимальный вес на выходных
            var max = netResult.Max();
            if (max > 0) //если это не нуль, значит хоть что-то да распозналось
            {
                //если таких (максимальных) весов больше одного, значит, сеть не смогла точно идентифицировать символ
                if (netResult.Count(j => Math.Abs(j - max) < float.Epsilon) == 1)
                {
                    //таки смогла, присваиваем результат
                    result = netResult.ToList().FindIndex(j => Math.Abs(j - max) < float.Epsilon);
                }
            }

            return result;
        }

    }
}