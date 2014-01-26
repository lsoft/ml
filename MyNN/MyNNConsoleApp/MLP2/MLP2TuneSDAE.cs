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
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Default;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2TuneSDAE
    {
        public static void Tune()
        {
            var rndSeed = 38779;
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
                "SDAE20140113100839 MLP2/mlp20140114000706.mynn");
                //"MLP20131218124915/epoche 42/20131219100700-perItemError=3,6219.mynn");
                //"MLP20131219184828/epoche 28/20131220091649-perItemError=2,600619.mynn");

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());


            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.001f, 0.99f),
                    1,
                    0.0f,
                    50,
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
                            new CPUBackpropagationAlgorithm(
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider),
                        mlp,
                        validation,
                        config);

                //var noiser = new SetOfNoisers(
                //    randomizer,
                //    new Pair<float, INoiser>(0.25f, new ZeroMaskingNoiser(randomizer, 0.25f)),
                //    new Pair<float, INoiser>(0.25f, new SaltAndPepperNoiser(randomizer, 0.25f)),
                //    new Pair<float, INoiser>(0.25f, new GaussNoiser(0.2f, false)),
                //    new Pair<float, INoiser>(0.25f, new MultiplierNoiser(randomizer, 1f))
                //    );

                //var noiser = new SetOfNoisers2(
                //    randomizer,
                //    new Pair<float, INoiser>(0.25f, new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))),
                //    new Pair<float, INoiser>(0.25f, new SaltAndPepperNoiser(randomizer, 0.25f, new RandomRange(randomizer))),
                //    new Pair<float, INoiser>(0.25f, new GaussNoiser(0.2f, false, new RandomRange(randomizer))),
                //    new Pair<float, INoiser>(0.25f, new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)))
                //    );

                //var noiser = new AllNoisers(
                //    randomizer,
                //    new ZeroMaskingNoiser(randomizer, 0.15f, new RandomRange(randomizer)),
                //    new SaltAndPepperNoiser(randomizer, 0.15f, new RandomRange(randomizer)),
                //    new GaussNoiser(0.10f, false, new RandomRange(randomizer)),
                //    new MultiplierNoiser(randomizer, 0.3f, new RandomRange(randomizer)),
                //    new DistanceChangeNoiser(randomizer, 0.3f, 2, new RandomRange(randomizer))
                //    );

                //var noiser = new AllNoisers(
                //    randomizer,
                //    new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer)),
                //    new SaltAndPepperNoiser(randomizer, 0.25f, new RandomRange(randomizer)),
                //    new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                //    new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                //    new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer))
                //    );

                var noiser = new SetOfNoisers(
                    randomizer,
                    new Pair<float, INoiser>(0.33f, new ZeroMaskingNoiser(randomizer, 0.25f)),
                    new Pair<float, INoiser>(0.33f, new SaltAndPepperNoiser(randomizer, 0.25f)),
                    new Pair<float, INoiser>(0.34f, new GaussNoiser(0.20f, false))
                    );

                //обучение сети
                alg.Train(
                    //new NoDeformationTrainDataProvider(trainData.ConvertToAutoencoder()).GetDeformationDataSet);
                    new NoiseDataProvider(trainData.ConvertToAutoencoder(), noiser).GetDeformationDataSet);
            }


        }
    }
}
