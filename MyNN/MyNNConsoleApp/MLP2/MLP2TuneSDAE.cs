using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Metadata.W3cXsd2001;
using System.Text;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2TuneSDAE
    {
        public static void Tune()
        {
            var rndSeed = 387781;
            var randomizer = new DefaultRandomizer(++rndSeed);

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
                "SDAE20140130152633 MLP2/mlp20140131053808.sdae");
                //"MLP20131218124915/epoche 42/20131219100700-perItemError=3,6219.mynn");
                //"MLP20131219184828/epoche 28/20131220091649-perItemError=2,600619.mynn");

            Console.WriteLine("Network configuration: " + mlp.GetLayerInformation());


            using (var clProvider = new CLProvider())
            {
                const int epocheCount = 50;

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.001f, 0.99f),
                    1,
                    0.0f,
                    epocheCount,
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
                        new CPUBackpropagationEpocheTrainer(
                            VectorizationSizeEnum.VectorizationMode16,
                            mlp,
                            config,
                            clProvider),
                        mlp,
                        validation,
                        config);

                Func<int, INoiser> noiserProvider =
                    (int epocheNumber) =>
                    {
                        if (epocheCount == epocheNumber)
                        {
                            return
                                new NoNoiser();
                        }

                        var coef = (epocheCount - epocheNumber) / (float)epocheCount;

                        var noiser = new AllNoisers(
                            randomizer,
                            new GaussNoiser(coef * 0.20f, false, new RandomRange(randomizer)),
                            new MultiplierNoiser(randomizer, coef * 1f, new RandomRange(randomizer)),
                            new DistanceChangeNoiser(randomizer, coef * 1f, 3, new RandomRange(randomizer)),
                            new SaltAndPepperNoiser(randomizer, coef * 0.1f, new RandomRange(randomizer)),
                            new ZeroMaskingNoiser(randomizer, coef * 0.25f, new RandomRange(randomizer))
                            );

                        return noiser;
                    };


                //var noiser = new AllNoisers(
                //    randomizer,
                //    new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                //    new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                //    new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                //    new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                //    new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                //    );

                //обучение сети
                alg.Train(
                    new NoiseDataProvider(trainData.ConvertToAutoencoder(), noiserProvider));
            }


        }
    }
}
