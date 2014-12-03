using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.CTRP
{
    public class TrainMLP
    {
        public static void DoTrain()
        {
            var randomizer = new DefaultRandomizer(2387184);

            var trainDataSetProvider = GetTrainProvider(
                300000,//10000
                false,
                false
                );
            
            var validationData = GetValidation(
                100000,//10000
                false,
                false
                );

            var serialization = new SerializationHelper();

            //string filepath = null;
            //using (var ofd = new OpenFileDialog())
            //{
            //    ofd.InitialDirectory = Directory.GetCurrentDirectory();

            //    ofd.Multiselect = false;

            //    if (ofd.ShowDialog() != DialogResult.OK)
            //    {
            //        return;
            //    }

            //    filepath = ofd.FileNames[0];
            //}

            //var mlp = serialization.LoadFromFile<MLP>(
            //    filepath
            //    );

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var mlp = mlpFactory.CreateMLP(
                string.Format(
                    "mlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss")),
                new IFunction[]
                {
                    null,
                    new RLUFunction(), 
                    new SigmoidFunction(1f), 
                },
                new int[]
                {
                    validationData.InputLength,
                    400,
                    validationData.OutputLength
                });

            ConsoleAmbientContext.Console.WriteLine("Created " + mlp.GetLayerInformation());

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new Loglikelihood(), 
                    validationData), 
                null
                );

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(true), false))
            {
                const int epocheCount = 50;

                //0.433
                //var config = new LearningAlgorithmConfig(
                //    new HalfSquaredEuclidianDistance(), 
                //    new LinearLearningRate(0.02f, 0.99f),
                //    1,
                //    0f,
                //    epocheCount,
                //    -1f,
                //    -1f
                //    );

                //0.432
                var config = new LearningAlgorithmConfig(
                    new Loglikelihood(),
                    new LinearLearningRate(0.00002f, 0.99f),
                    10,
                    0.001f,
                    epocheCount,
                    -1f,
                    -1f
                    );

                var mlpFolderName = string.Format(
                    "mlp{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlpContainer = rootContainer.GetChildContainer(mlpFolderName);

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
            int desiredCount,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new CRPDataItemLoader(
                "DATA #1",
                "__train.bin",
                desiredCount,
                dataItemFactory
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iteratorFactory = new CacheDataIteratorFactory(
                100,
                new DataIteratorFactory()
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
            int desiredCount,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new CRPDataItemLoader(
                "DATA #1",
                "__validation.bin",
                desiredCount,
                dataItemFactory
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iteratorFactory = new CacheDataIteratorFactory(
                100,
                new DataIteratorFactory()
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation();
                });

            var validationDataSet = new DataSet(
                iteratorFactory,
                itemTransformationFactory,
                dataItemLoader,
                0
                );

            return
                validationDataSet;
        }

    }
}
