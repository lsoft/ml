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
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment4
    {
        public static void Execute()
        {
            var rndSeed = 895788;
            var randomizer = 
                new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //1
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //10
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            const float p = 0.5f;

            const int sampleCount = 2500;

            var folderName = "_DropConnectMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

            var mlp = new MLP(
                randomizer,
                null,
                folderName,
                new IFunction[]
                {
                    null,
                    new HyperbolicTangensFunction(), 
                    new HyperbolicTangensFunction(), 
                    new SigmoidFunction(1f), 
                },
                new int[]
                {
                    784,
                    800,
                    800,
                    10
                });


            //var mlp = SerializationHelper.LoadFromFile<MLP>(
            //    "_DropConnectMLP20131215225141/epoche 97/20131217064313-9851 out of 98,51%.mynn");

            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.02f, 0.997f),
                    1,
                    0.0f,
                    1000,
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

            Console.WriteLine("Experiment #4 finished");
            Console.ReadLine();
        }
    }
}
