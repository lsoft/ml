using System;
using System.Collections.Generic;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using MyNN.Boosting.SAMMEBoosting.EpocheTrainers.Classifiers;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train;
using MyNN.NeuralNet.Train.Algo;
using MyNN.NeuralNet.Train.Metrics;

namespace MyNN.Boosting.SAMMEBoosting.EpocheTrainers
{
    public class MLP_CRL_Trainer : IEpocheTrainer
    {
        private int _rndSeed;
        private readonly int _firstLayerNonBiasNeuronCount;
        private readonly int _hiddenLayerNonBiasNeuronCount;
        private readonly int _classesCount;

        private MultiLayerNeuralNetwork _net;
        private DataSet _trainData;
        private DataSet _validationData;

        private int _bestCorrectCount;
        private float _bestCumulativeError;
        private MultiLayerNeuralNetwork _bestNetwork;

        public MLP_CRL_Trainer(
            int rndSeed,
            int firstLayerNonBiasNeuronCount,
            int hiddenLayerNonBiasNeuronCount,
            int classesCount)
        {
            _rndSeed = rndSeed;
            _firstLayerNonBiasNeuronCount = firstLayerNonBiasNeuronCount;
            _hiddenLayerNonBiasNeuronCount = hiddenLayerNonBiasNeuronCount;
            _classesCount = classesCount;
        }

        public IEpocheClassifier TrainEpocheClassifier(
            List<double[]> epocheInputs, 
            List<int> epocheLabels,
            int outputLength,
            int inputLength)
        {
            _net = new MultiLayerNeuralNetwork(
                        null,
                        null,
                        new IFunction[]
                        {
                            null,
                            new SigmoidFunction(1),
                            new SigmoidFunction(1),
                        },
                        ref _rndSeed,
                        _firstLayerNonBiasNeuronCount, _hiddenLayerNonBiasNeuronCount, _classesCount);
            StoreBestNetwork(_net);

            _bestCorrectCount = int.MinValue;
            _bestCumulativeError = float.MaxValue;

            //конструируем датасет для обучения MLP
            _trainData = new DataSet();
            for (var cc = 0; cc < epocheInputs.Count; cc++)
            {
                var input = epocheInputs[cc].ToList().ConvertAll(j => (float) j).ToArray();

                var output = new float[outputLength];
                output[epocheLabels[cc]] = 1f;

                _trainData.AddItem(
                    new DataItem(
                        input,
                        output));
            }
            _validationData = new DataSet(_trainData);

            //размножаем
            _trainData.ExpandDataSet(
                0.1f,
                30,
                ref _rndSeed);

            var conf = new LearningAlgorithmConfig(
                new ConstLearningRate(0.1f),
                1,
                0.0f,
                100,
                0.0001f,
                -1.0f,
                new HalfSquaredEuclidianDistance());

            using (var universe = new VNNCLProvider(_net))
            {
                //создаем объект просчитывающий сеть
                var computer =
                    new VOpenCLComputer(universe, true);

                _net.SetComputer(computer);

                var alg =
                    new VOpenCLBackpropAlgorithm(
                        _net,
                        conf,
                        Validation,
                        universe);

                //валидация используемой сети
                Console.WriteLine("Default net validation results:");
                Validation(_net, "_pretrain", float.MaxValue, false);

                //обучение сети
                alg.Train(
                    (int epocheNumber) =>
                    {
                        return _trainData;
                    });
            }

            return
                new MLPClassifier(_bestNetwork);
        }

        public void Validation(
            MultiLayerNeuralNetwork network,
            string epocheRoot,
            float cumulativeError,
            bool allowToSave)
        {
            var correctArray = new int[_validationData[0].OutputLength];//уродливо!!!
            var totalArray = new int[_validationData[0].OutputLength];//уродливо!!!

            var totalCorrectCount = 0;
            var totalFailCount = 0;

            var netResults = network.ComputeOutput(_validationData.GetInputPart());
            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];
                var correctIndex = testItem.OutputIndex;

                var success = false;

                //берем максимальный вес на выходных
                var max = netResult.Max();
                if (max > 0) //если это не нуль, значит хоть что-то да распозналось
                {
                    //если таких (максимальных) весов больше одного, значит, сеть не смогла точно идентифицировать символ
                    if (netResult.Count(j => Math.Abs(j - max) < float.Epsilon) == 1)
                    {

                        //таки смогла, присваиваем результат
                        var recognizeIndex = netResult.ToList().FindIndex(j => Math.Abs(j - max) < float.Epsilon);

                        success = correctIndex == recognizeIndex;
                    }
                }

                totalArray[correctIndex]++;

                if (success)
                {
                    totalCorrectCount++;
                    correctArray[correctIndex]++;
                }
                else
                {
                    totalFailCount++;
                }
            }

            var totalCount = totalCorrectCount + totalFailCount;

            var correctPercentCount = ((int)(totalCorrectCount * 10000 / totalCount) / 100.0);

            Console.WriteLine(
                string.Format(
                    "Success: {0}, fail: {1}, total: {2}, success %: {3}",
                    totalCorrectCount,
                    totalFailCount,
                    totalCount,
                    correctPercentCount));

            //по классам репортим
            if (correctArray.Length < 100)
            {
                for (var cc = 0; cc < correctArray.Length; cc++)
                {
                    Console.Write(cc + ": ");

                    var p = (int)((correctArray[cc] / (double)totalArray[cc]) * 10000) / 100.0;
                    Console.Write(correctArray[cc] + "/" + totalArray[cc] + "(" + p + "%)");

                    if (cc != (correctArray.Length - 1))
                    {
                        Console.Write("   ");
                    }
                    else
                    {
                        Console.WriteLine(string.Empty);
                    }
                }

                //определяем классы, в которых нуль распознаваний
                var zeroClasses = string.Empty;
                var index = 0;
                foreach (var c in correctArray)
                {
                    if (c == 0)
                    {
                        zeroClasses += index;
                    }

                    index++;
                }

                if (!string.IsNullOrEmpty(zeroClasses))
                {
                    Console.WriteLine("Not recognized: " + zeroClasses);
                }
            }
            else
            {
                Console.WriteLine("Too many domains, details output is disabled");
            }

            //если результат лучше, чем был, то сохраняем его
            if ((_bestCorrectCount <= totalCorrectCount && totalCorrectCount > 0) || (correctArray.Length >= 100 && _bestCumulativeError >= cumulativeError))
            {
                _bestCorrectCount = Math.Max(_bestCorrectCount, totalCorrectCount);
                _bestCumulativeError = Math.Min(_bestCumulativeError, cumulativeError);

                if (allowToSave)
                {
                    //SerializationHelper.SaveToFile(
                    //    network,
                    //    string.Format(
                    //        "{0}-{1}-{2}%-err={3}.mynn",
                    //        DateTime.Now.ToString("yyyyMMddHHmmss"),
                    //        totalCorrectCount,
                    //        correctPercentCount,
                    //        cumulativeError));

                    StoreBestNetwork(network);

                    Console.WriteLine("Stored!");
                }
            }
        }

        private void StoreBestNetwork(MultiLayerNeuralNetwork network)
        {
            _bestNetwork = SerializationHelper.DeepClone(network);
            _bestNetwork.SetComputer(new DefaultComputer(_bestNetwork));
        }
    }
}