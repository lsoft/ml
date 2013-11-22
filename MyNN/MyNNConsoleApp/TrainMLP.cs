using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train.Algo;
using MyNN.NeuralNet.Train.Metrics;
using MyNN.NeuralNet.Train.Validation;

namespace MyNNConsoleApp
{
    public class TrainMLP
    {
        public void Train(
            DataSet trainData,
            DataSet validationData)
        {

            var rndSeed = 74;

            var mlp = SerializationHelper.LoadFromFile<MultiLayerNeuralNetwork>(
                //"MLP20131013110356/epoche 12/20131013124252-err=0,1326502.mynn");
                "SDAE20131031190905ZMN0.25/mlp20131031215516.mynn");

            mlp.AutoencoderCut();

            mlp.AddLayer(
                new SigmoidFunction(1f),
                24,
                false,
                ref rndSeed);

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());

            using (var universe = new VNNCLProvider(mlp))
            {
                //создаем объект просчитывающий сеть
                var computer =
                    new VOpenCLComputer(universe, true);
                    //new DefaultComputer(mlp);

                mlp.SetComputer(computer);

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.01f, 0.99f),
                    1,
                    0.0f,
                    1000,
                    0.0001f,
                    -1.0f,
                    new HalfSquaredEuclidianDistance());

                var alg =
                    //new OpenCLNaiveBackpropAlgorithm(
                    new VOpenCLBackpropAlgorithm(
                        mlp,
                        config,
                        new MetricErrorValidation(
                            new RMSE(), 
                            validationData, 
                            300, 
                            100).Validate,
                        universe);

                //обучение сети
                alg.Train(
                    //new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
                    new NoiseDataProvider(trainData, new GaussNoiser(0.025f, false)).GetDeformationDataSet);
            }


            return;
        }
    }
}
