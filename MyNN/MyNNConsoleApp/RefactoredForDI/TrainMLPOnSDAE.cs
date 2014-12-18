using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MyNN;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.MNIST;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Backpropagation.Validation.Drawer.Factory;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.GPU;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainMLPOnSDAE
    {
        public static void DoTrain()
        {
            const int trainMaxCountFilesInCategory = 1000;
            const int validationMaxCountFilesInCategory = 300;

            var trainDataSetProvider = GetTrainProvider(
                trainMaxCountFilesInCategory,
                false,
                true
                );
            
            var validationData = GetValidation(
                validationMaxCountFilesInCategory,
                false,
                true
                );

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

            var mlp = serialization.LoadFromFile<MLP>(
                filepath
                );

            ConsoleAmbientContext.Console.WriteLine("Loaded " + mlp.GetLayerInformation());

            mlp.AutoencoderCutTail();

            mlp.AddLayer(
                new SigmoidFunction(1f),
                10,
                false);

            //for (var layerIndex = 2; layerIndex < mlp.Layers.Length; layerIndex++)
            //{
            //    var l = mlp.Layers[layerIndex];

            //    for (var neuronIndex = 0; neuronIndex < l.NonBiasNeuronCount; neuronIndex++)
            //    {
            //        var n = l.Neurons[neuronIndex];

            //        for (var weightIndex = 0; weightIndex < n.Weights.Length; weightIndex++)
            //        {
            //            n.Weights[weightIndex] *= 0.5f;
            //        }
            //    }
            //}

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                new GridReconstructDrawerFactory(
                    new MNISTVisualizerFactory(), 
                    validationData,
                    300
                    )
                );

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(true), false))
            {
                var mlpName = string.Format(
                    "mlp{0}.basedonsdae.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                const int epocheCount = 30;

                var config = new LearningAlgorithmConfig(
                    new HalfSquaredEuclidianDistance(), 
                    new LinearLearningRate(0.02f, 0.99f),
                    1,
                    0.001f,
                    epocheCount,
                    -1f,
                    -1f
                    );

                var mlpContainer = rootContainer.GetChildContainer(mlpName);

                var mlpContainerHelper = new MLPContainerHelper();

                var algo = new Backpropagation(
                    new GPUEpocheTrainer(
                        mlp,
                        config,
                        clProvider),
                    mlpContainerHelper,
                    mlpContainer,
                    mlp,
                    validation,
                    config
                    );

                algo.Train(
                    trainDataSetProvider
                    );
            }
        }

        private static IDataSetProvider GetTrainProvider(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoader(
                "_MNIST_DATABASE/mnist/trainingset/",
                maxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iteratorFactory = new DataIteratorFactory(
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation();
                });

            var dataSetFactory = new DataSetFactory(
                iteratorFactory,
                itemTransformationFactory
                );

            var dataSetProvider = new DataSetProvider(
                dataSetFactory,
                dataItemLoader
                );

            return
                dataSetProvider;
        }

        private static IDataSet GetValidation(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoader(
                "_MNIST_DATABASE/mnist/testset/",
                maxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iterationFactory = new DataIteratorFactory(
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation();
                });

            var validationDataSet = new DataSet(
                iterationFactory,
                itemTransformationFactory,
                dataItemLoader,
                0
                );

            return
                validationDataSet;
        }

    }
}
