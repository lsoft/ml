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
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.GPU.Default;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp.ClassificationAutoencoder
{
    public class CATuneSCDAE
    {
        public static void Tune()
        {
            var rndSeed = 12323235;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //10
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

            var mlp = SerializationHelper.LoadFromFile<MLP>(
                "SCDAE20140127180216/mlp20140128223317.scdae");
                //"MLP20131218124915/epoche 42/20131219100700-perItemError=3,6219.mynn");
                //"MLP20131219184828/epoche 28/20131220091649-perItemError=2,600619.mynn");

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());


            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(), false))
            {
                const int epocheCount = 50;

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.01f, 0.99f),
                    1,
                    0.0f,
                    epocheCount,
                    0.0001f,
                    -1.0f);

                var validation = new ClassificationAutoencoderValidation(
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

                Func<int, INoiser> noiserProvider =
                    (int epocheNumber) =>
                    {
                        var coef = (epocheCount - epocheNumber)/(float) epocheCount;

                        var noiser = new AllNoisers(
                            randomizer,
                            new ZeroMaskingNoiser(randomizer, coef * 0.20f, new RandomRange(randomizer)),
                            new SaltAndPepperNoiser(randomizer, coef * 0.20f, new RandomRange(randomizer)),
                            new GaussNoiser(coef * 0.175f, false, new RandomRange(randomizer)),
                            new MultiplierNoiser(randomizer, coef * 0.75f, new RandomRange(randomizer)),
                            new DistanceChangeNoiser(randomizer, coef * 1f, 3, new RandomRange(randomizer))
                            );

                        return noiser;
                    };


                //обучение сети
                alg.Train(
                    //new NoDeformationTrainDataProvider(trainData.ConvertToAutoencoder()).GetDeformationDataSet);
                    new NoiseDataProvider(trainData.ConvertToClassificationAutoencoder(), noiserProvider).GetDeformationDataSet);
            }
        }



    }
}
