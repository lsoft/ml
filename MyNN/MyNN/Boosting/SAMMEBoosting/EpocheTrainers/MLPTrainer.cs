using System;
using System.Collections.Generic;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using MyNN.Boosting.SAMMEBoosting.EpocheTrainers.Classifiers;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL;

namespace MyNN.Boosting.SAMMEBoosting.EpocheTrainers
{
    public class MLPTrainer : IEpocheTrainer, IValidation
    {
        private readonly IRandomizer _randomizer;
        private readonly int _firstLayerNonBiasNeuronCount;
        private readonly int _hiddenLayerNonBiasNeuronCount;
        private readonly int _classesCount;

        private MLP _net;
        private DataSet _trainData;
        private DataSet _validationData;

        private int _bestCorrectCount;
        private float _bestTotalError;
        private MLP _bestNetwork;

        public MLPTrainer(
            IRandomizer randomizer,
            int firstLayerNonBiasNeuronCount,
            int hiddenLayerNonBiasNeuronCount,
            int classesCount)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
            _firstLayerNonBiasNeuronCount = firstLayerNonBiasNeuronCount;
            _hiddenLayerNonBiasNeuronCount = hiddenLayerNonBiasNeuronCount;
            _classesCount = classesCount;

            throw new NotImplementedException("Не проверено после глубокого рефакторинга! Прежде чем юзать в бою - необходимо проверить");
        }

        public IEpocheClassifier TrainEpocheClassifier(
            List<double[]> epocheInputs,
            List<int> epocheLabels,
            int outputLength,
            int inputLength)
        {
            _net = new MLP(
                _randomizer,
                null,
                null,
                new IFunction[]
                {
                    null,
                    new SigmoidFunction(1),
                    new SigmoidFunction(1),
                },
                _firstLayerNonBiasNeuronCount, _hiddenLayerNonBiasNeuronCount, _classesCount);

            StoreBestNetwork(_net);

            _bestCorrectCount = int.MinValue;
            _bestTotalError = float.MaxValue;

            //конструируем датасет для обучения MLP
            _trainData = new DataSet();
            for (var cc = 0; cc < epocheInputs.Count; cc++)
            {
                var input = epocheInputs[cc].ToList().ConvertAll(j => (float)j).ToArray();

                var output = new float[outputLength];
                output[epocheLabels[cc]] = 1f;

                _trainData.AddItem(
                    new DataItem(
                        input,
                        output));
            }
            _validationData = _trainData;

            var conf = new LearningAlgorithmConfig(
                new ConstLearningRate(0.07f),
                1,
                0.0f,
                200,
                0.0001f,
                -1.0f);

            using (var clProvider = new CLProvider())
            {
                //создаем объект просчитывающий сеть
                var alg =
                    new BackpropagationAlgorithm(
                        _randomizer,
                        (currentMLP, currentConfig) =>
                            new OpenCLBackpropagationAlgorithm(
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider),
                        _net,
                        this,
                        conf);

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

        public float Validate(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave)
        {
            var correctArray = new int[_validationData[0].OutputLength];//уродливо!!!
            var totalArray = new int[_validationData[0].OutputLength];//уродливо!!!

            var totalCorrectCount = 0;
            var totalFailCount = 0;

            var errorMetrics = new RMSE();

            var netResults = forwardPropagation.ComputeOutput(_validationData);

            var totalError = 0f;

            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];

                #region суммируем ошибку

                var err = errorMetrics.Calculate(
                    netResult.State,
                    testItem.Output);

                totalError += err;

                #endregion

                #region вычисляем успешность классификации

                var correctIndex = testItem.OutputIndex;

                var success = false;

                //берем максимальный вес на выходных
                var max = netResult.State.Max();
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

                #endregion
            }


            var perItemError = totalError / _validationData.Count;
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
            if ((_bestCorrectCount <= totalCorrectCount && totalCorrectCount > 0) || (correctArray.Length >= 100 && _bestTotalError >= totalError))
            {
                _bestCorrectCount = Math.Max(_bestCorrectCount, totalCorrectCount);
                _bestTotalError = Math.Min(_bestTotalError, totalError);

                if (allowToSave)
                {
                    StoreBestNetwork(forwardPropagation.MLP);

                    Console.WriteLine("Stored!");
                }
            }

            return perItemError;
        }

        private void StoreBestNetwork(MLP network)
        {
            _bestNetwork = new SerializationHelper().DeepClone(network);
        }
    }
}