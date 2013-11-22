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
    public class TrainAutoencoderNLNCA
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

            //создаем объект просчитывающий сеть
            var computer =
                new DefaultComputer(network);

            network.SetComputer(computer);

            var batchSize = 100;//trainData.Count/12;

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.1f, 0.99f),
                batchSize,
                0.0f,
                1000,
                0.0001f,
                -1.0f,
                new HalfSquaredEuclidianDistance());

            var alg =
                //new NaiveBackpropagationLearningAlgorithm(
                new NLNCAAutoencoderBackpropAlgorithm(
                    network,
                    config,
                    //new NLNCAValidation(trainData, validationData, colorProvider, 3).Validate,
                    new AutoencoderValidation(validationData, 100, 100).Validate,
                    (uzkii) => new DodfCalculatorVectorized(uzkii), 
                    0.1f,
                    50);

            //обучение сети
            alg.Train(new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);


            return;
        }
    }
}
