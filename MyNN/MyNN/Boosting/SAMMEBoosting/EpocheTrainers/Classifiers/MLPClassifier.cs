using System;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using MyNN.NeuralNet.Structure;

namespace MyNN.Boosting.SAMMEBoosting.EpocheTrainers.Classifiers
{
    internal class MLPClassifier : IEpocheClassifier
    {
        private readonly MultiLayerNeuralNetwork _network;

        public MLPClassifier(MultiLayerNeuralNetwork network)
        {
            if (network == null)
            {
                throw new ArgumentNullException("network");
            }

            _network = network;
        }

        public int Compute(double[] input)
        {
            var inputf = input.ToList().ConvertAll(j => (float) j).ToArray();

            var netResult = _network.ComputeOutput(inputf);

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