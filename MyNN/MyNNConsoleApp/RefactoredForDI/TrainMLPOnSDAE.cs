using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
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
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainMLPOnSDAE
    {
        public static void DoTrain()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

            var serialization = new SerializationHelper();

            var mlp = serialization.LoadFromFile<MLP>(
                "ae20140821114722.ae/epoche 18/sdae2d20140820001058.sdae");

            mlp.AutoencoderCutTail();

            mlp.AddLayer(
                new SigmoidFunction(1f),
                10,
                false);

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                new GridReconstructDrawer(
                    new MNISTVisualizer(), 
                    validationData,
                    300,
                    100)
                );

            using (var clProvider = new CLProvider())
            {
                var mlpName = string.Format(
                    "justmlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.02f, 0.99f),
                    1,
                    0f,
                    25,
                    -1f,
                    -1f
                    );

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoDeformationTrainDataProvider(trainData)
                        );

                var mlpContainer = rootContainer.GetChildContainer(mlpName);

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
                    config
                    );

                algo.Train(
                    trainDataProvider
                    );
            }
        }
    }
}
