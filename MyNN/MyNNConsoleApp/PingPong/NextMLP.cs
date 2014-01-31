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
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.PingPong
{
    public class NextMLP
    {
        public static void Execute()
        {
            var rndSeed = 123456;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //100
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            //через обученную сеть генерируем данные для следующей эпохи
            var mlpPath = "PingPong/Experiment0/Step0/20140115132416-9876 out of 98,76%.mynn";

            DataSet trainNext;
            DataSet validationNext;
            NextDataSet.NextDataSets(mlpPath, trainData, validationData, out trainNext, out validationNext);

            //обучаем вторичный каскад

            var mlp_classifier = SerializationHelper.LoadFromFile<MLP>(
                "PingPong/Experiment0/MLP20140115135217/epoche 42/20140115172111-perItemError=3,886454.mynn");
            mlp_classifier.SetRootFolder("PingPong/Experiment0");

            mlp_classifier.AutoencoderCutTail();

            mlp_classifier.AddLayer(
                new SigmoidFunction(1f),
                10,
                false);

            Console.WriteLine("Network configuration: " + mlp_classifier.DumpLayerInformation());

            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.02f, 0.98f),
                    1,
                    0.0f,
                    50,
                    0.0001f,
                    -1.0f);

                var validation = new ClassificationValidation(
                    new FileSystemMLPSaver(serialization),
                    new HalfSquaredEuclidianDistance(),
                    validationNext,
                    300,
                    100);

                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new CPUBackpropagationAlgorithm(
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider),
                        mlp_classifier,
                        validation,
                        config);

                //обучение сети
                alg.Train(
                    new NoDeformationTrainDataProvider(trainNext).GetDeformationDataSet);
            }
        }
    }
}
