using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train.Algo;
using MyNN.NeuralNet.Train.Algo.NLNCA;
using MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator;
using MyNN.NeuralNet.Train.Metrics;
using MyNN.NeuralNet.Train.Validation;
using MyNN.NeuralNet.Train.Validation.NLNCA;

namespace MyNNConsoleApp
{
    public class TrainNLNCA
    {
        public void Train(
            ref int rndSeed,
            MultiLayerNeuralNetwork network,
            DataSet trainData,
            DataSet validationData,
            IColorProvider colorProvider)
        {
            if (network == null)
            {
                throw new ArgumentNullException("network");
            }
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (colorProvider == null)
            {
                throw new ArgumentNullException("colorProvider");
            }

            network.DumpLayerInformation();

            using (var universe = new VNNCLProvider(network))
            {
                //создаем объект просчитывающий сеть
                var computer =
                    new VOpenCLComputer(universe, true);

                network.SetComputer(computer);

                var batchSize = trainData.Count/12;

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(5.0f, 0.99f),
                    batchSize,
                    0.0f,
                    1000,
                    0.0001f,
                    -1.0f,
                    new HalfSquaredEuclidianDistance());

                var alg =
                    //new VOpenCLBackpropAlgorithm(
                    new VOpenCLNLNCABackpropAlgorithm(
                        network,
                        config,
                        new NLNCAValidation(trainData, validationData, colorProvider, 3).Validate,
                        (uzkii) => new DodfCalculatorVectorized(uzkii),
                        universe,
                        null);

                //обучение сети
                alg.Train(new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
            }


            return;
        }

    }
}
