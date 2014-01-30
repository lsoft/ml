using System;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Default;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.GPU.Default;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp.ClassificationAutoencoder
{
    public class CATrainMLPBasedOnSCDAE
    {
        public static void Tune()
        {
            var rndSeed = 18892;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //1000
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            var mlp = SerializationHelper.LoadFromFile<MLP>(
                //"SCDAE20140117102818/mlp20140118003337.mynn");
                "MLP20140129105235/epoche 49/20140130022618-(1,195425) 9883 correct out of 10000 - 98,83%.mynn");

            mlp.AutoencoderCutTail();

            mlp.AddLayer(
                new SigmoidFunction(1f),
                //new IRLUFunction(), 
                10,
                false);

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());


            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(), false))
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
                    validationData,
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
                        mlp,
                        validation,
                        config);

                //обучение сети
                alg.Train(
                    new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
                    //new NoiseDataProvider(trainData, noiser).GetDeformationDataSet);
            }


        }
    }
}
