using System;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.DropConnectInference
{
    public class TuneSDAE
    {
        public static void Tune()
        {
            var rndSeed = 88178;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //100
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
                //"SDAE20131229105634 MLP2/mlp20131230080924.mynn");
                "SDAE20131231204527 MLP2/mlp20140101203652.mynn");

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());


            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.001f, 0.99f),
                    1,
                    0.0f,
                    100,
                    0.0001f,
                    -1.0f);

                var validation = new AutoencoderValidation(
                    new FileSystemMLPSaver(serialization),
                    new HalfSquaredEuclidianDistance(), 
                    validationData.ConvertToAutoencoder(),
                    300,
                    100);

                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new DropConnectBitOpenCLBackpropagationAlgorithm<OpenCLLayerInferenceNew16>(
                                randomizer,
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider,
                                2500,
                                0.5f),
                        mlp,
                        validation,
                        config);

                var noiser = new AllNoisers(
                    randomizer,
                    new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer)),
                    new SaltAndPepperNoiser(randomizer, 0.25f, new RandomRange(randomizer)),
                    new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                    new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                    new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer))
                    );

                //обучение сети
                alg.Train(
                    new NoiseDataProvider(trainData.ConvertToAutoencoder(), noiser).GetDeformationDataSet);
            }


        }
    }
}
