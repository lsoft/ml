using System;
using System.IO;
using System.Security.Permissions;
using System.Windows.Forms;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataLoader;

using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.LearningRateController;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TuneSDAE_Dropout
    {
        public static void Tune()
        {
            var rndSeed = 61962;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var dataItemFactory = new DataItemFactory();

            var toa = new ToAutoencoderDataSetConverter(
                dataItemFactory
                );

            var trainData = MNISTDataLoader.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue,
                false,
                dataItemFactory
                );
            trainData.Normalize();
            trainData = toa.Convert(trainData);

            var validationData = MNISTDataLoader.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue,
                false,
                dataItemFactory
                );
            validationData.Normalize();
            validationData = toa.Convert(validationData);

            var serialization = new SerializationHelper();

            string filepath = null;
            using (var ofd = new OpenFileDialog())
            {
                ofd.InitialDirectory = Directory.GetCurrentDirectory();

                ofd.Multiselect = false;

                if (ofd.ShowDialog() != DialogResult.OK)
                {
                    return;
                }

                filepath = ofd.FileNames[0];
            }

            var mlp = new SerializationHelper().LoadFromFile<MLP>(
                filepath
                );

            Console.WriteLine("Network configuration: " + mlp.GetLayerInformation());


            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(true), false))
            {
                const int epocheCount = 250;

                var config = new LearningAlgorithmConfig(
                    new HalfSquaredEuclidianDistance(),
                    new LinearLearningRate(0.001f, 0.99f),
                    1,
                    0.001f,
                    epocheCount,
                    0.0001f,
                    -1.0f);

                var rootContainer = new FileSystemArtifactContainer(
                    ".",
                    serialization);

                var sdaeName = string.Format(
                    "sdae{0}.tuned.dropout.sdae",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlpContainer = rootContainer.GetChildContainer(
                    sdaeName);

                var mlpContainerHelper = new MLPContainerHelper();

                var validation = new Validation(
                    new MetricsAccuracyCalculator(
                        new HalfSquaredEuclidianDistance(),
                        validationData),
                    new GridReconstructDrawer(
                        new MNISTVisualizer(),
                        validationData,
                        300,
                        100)
                    );

                var maskContainerFactory = new BigArrayMaskContainerFactory(
                    randomizer,
                    clProvider
                    );

                const float p = 0.5f;

                var alg =
                    new Backpropagation(
                        new GPUDropoutEpocheTrainer(
                            randomizer,
                            maskContainerFactory,
                            mlp,
                            config,
                            clProvider,
                            p
                            ),
                        mlpContainerHelper,
                        mlpContainer,
                        mlp,
                        validation,
                        config);

                Func<int, INoiser> noiserProvider =
                    (int epocheNumber) =>
                    {
                        //if (epocheCount == epocheNumber)
                        //{
                        //    return
                        //        new NoNoiser();
                        //}

                        //var coef = (epocheCount - epocheNumber) / (float)epocheCount;
                        const float coef = 1f;

                        var noiser = new SequenceNoiser(
                            randomizer,
                            true,
                            new GaussNoiser(coef * 0.10f, false, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
                            new MultiplierNoiser(randomizer, coef * 0.7f, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
                            new DistanceChangeNoiser(randomizer, coef * 0.7f, 3, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
                            new SaltAndPepperNoiser(randomizer, coef * 0.1f, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
                            new ZeroMaskingNoiser(randomizer, coef * 0.25f, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength))
                            );

                        return noiser;
                    };

                //обучение сети
                alg.Train(
                    new NoiseDataProvider(trainData, noiserProvider, dataItemFactory));
            }


        }
    }
}
