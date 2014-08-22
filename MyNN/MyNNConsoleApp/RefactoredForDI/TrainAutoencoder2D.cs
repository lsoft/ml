using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.Drawer;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainAutoencoder2D
    {
        public static void DoTrain()
        {
            var toa = new ToAutoencoderDataSetConverter();

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                );
            trainData.Normalize();
            trainData = toa.Convert(trainData);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
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
                    new NoiseDataProvider(trainData, noiser2d)
                    );

            var mlp = serialization.LoadFromFile<MLP>(
                "ae20140821114722.ae/epoche 17/sdae2d20140820001058.sdae");
                //"sdae2d20140820001058.sdae");
                //"sdae2d20140820001058.sdae/sdae2d20140820001058.sdae");

            var autoencoderName = string.Format(
                "ae{0}.ae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlpContainer = rootContainer.GetChildContainer(autoencoderName);

            using (var clProvider = new CLProvider())
            {
                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    mlpContainer,
                    mlp,
                    validation,
                    config);

                algo.Train(trainDataProvider);
            }
        }
    }
}
