using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set.Item.Dense;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.Data.TypicalDataProvider;
using MyNN.Common.LearningRateController;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainAutoencoder2D
    {
        public static void DoTrain()
        {
            var dataItemFactory = new DenseDataItemFactory();

            var toa = new ToAutoencoderDataSetConverter(
                dataItemFactory);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue,
                true,
                dataItemFactory
                );
            trainData.Normalize();
            trainData = toa.Convert(trainData);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue,
                true,
                dataItemFactory
                );
            validationData.Normalize();
            validationData = toa.Convert(validationData);

            var randomizer = new DefaultRandomizer(123);

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

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

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(), 
                new LinearLearningRate(0.0007f, 0.99f),
                1,
                0f,
                50,
                -1f,
                -1f
                );

            var noiser2d = new SequenceNoiser(
                randomizer,
                false,
                new ElasticNoiser(randomizer, 100, 28, 28, true),
                new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer, trainData[0].InputLength))
                );
            //var noiser2d = new ElasticNoiser(randomizer, 100, 28, 28, true);

            var trainDataProvider = 
                new ConverterTrainDataProvider(
                    new ShuffleDataSetConverter(randomizer), 
                    new NoiseDataProvider(trainData, noiser2d, dataItemFactory)
                    );

            var mlp = serialization.LoadFromFile<MLP>(
                "ae20140821114722.ae/epoche 17/sdae2d20140820001058.sdae");
                //"sdae2d20140820001058.sdae");
                //"sdae2d20140820001058.sdae/sdae2d20140820001058.sdae");

            var autoencoderName = string.Format(
                "ae{0}.ae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlpContainer = rootContainer.GetChildContainer(autoencoderName);

            var mlpContainerHelper = new MLPContainerHelper();

            using (var clProvider = new CLProvider())
            {
                var algo = new Backpropagation(
                    new CPUEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    mlpContainerHelper,
                    mlpContainer,
                    mlp,
                    validation,
                    config);

                algo.Train(trainDataProvider);
            }
        }
    }
}
