using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnect;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment5
    {
        public static void Execute()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1
                );
            trainData.Normalize();
            trainData = new DataSet(
                new List<DataItem>
                {
                    trainData.Data[0],
                    //trainData.Data[1],
                },
                trainData.Visualizer);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10
                );
            validationData.Normalize();

            const float p = 0.5f;
            const int sampleCount = 10000;

            var serialization = new SerializationHelper();

            {
                var randomizer =
                    new NoRandomRandomizer();

                var folderName = "_DropConnectMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var mlp = new MLP(
                    randomizer,
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                    },
                        new int[]
                    {
                        784,
                        10
                    });


                using (var clProvider = new CLProvider())
                {
                    var config = new LearningAlgorithmConfig(
                        new ConstLearningRate(0.02f),
                        1,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

                    var validation = new ClassificationValidation(
                        new FileSystemMLPSaver(serialization),
                        new HalfSquaredEuclidianDistance(),
                        validationData,
                        300,
                        100);

                    var alg =
                        new BackpropagationAlgorithm(
                            randomizer,
                            (currentMLP, currentConfig) =>
                                new DropConnectCPUBackpropagationAlgorithm<VectorizedCPULayerInferenceV2>(
                                    randomizer,
                                    VectorizationSizeEnum.VectorizationMode16,
                                    currentMLP,
                                    currentConfig,
                                    clProvider,
                                    sampleCount,
                                    p),
                            mlp,
                            validation,
                            config);

                    //обучение сети
                    alg.Train(
                        new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
                }
            }

            {
                var randomizer =
                    new NoRandomRandomizer();

                var folderName = "_DropConnectMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var mlp = new MLP(
                    randomizer,
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f),
                    },
                    new int[]
                    {
                        784,
                        10
                    });


                using (var clProvider = new CLProvider())
                {
                    var config = new LearningAlgorithmConfig(
                        new ConstLearningRate(0.02f), 
                        1,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

                    var validation = new ClassificationValidation(
                        new FileSystemMLPSaver(serialization),
                        new HalfSquaredEuclidianDistance(),
                        validationData,
                        300,
                        100);

                    var alg =
                        new BackpropagationAlgorithm(
                            randomizer,
                            (currentMLP, currentConfig) =>
                                new DropConnectBitCPUBackpropagationAlgorithm<VectorizedCPULayerInferenceV2>(
                                    randomizer,
                                    VectorizationSizeEnum.VectorizationMode16,
                                    currentMLP,
                                    currentConfig,
                                    clProvider,
                                    sampleCount,
                                    p),
                            mlp,
                            validation,
                            config);

                    //обучение сети
                    alg.Train(
                        new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
                }
            }

            Console.WriteLine("Experiment #5 finished");
            Console.ReadLine();
        }
    }
}
