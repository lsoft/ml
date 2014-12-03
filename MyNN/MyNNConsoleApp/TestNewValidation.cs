using System;
using System.Collections.Generic;
using System.IO;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataLoader;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp
{
    internal class TestNewValidation
    {
        public static void DoTest()
        {

            if (File.Exists("grid.bmp"))
            {
                File.Delete("grid.bmp");
            }

            if (File.Exists("reconstruct.bmp"))
            {
                File.Delete("reconstruct.bmp");
            }

            {
                var validation = GetValidation(
                    int.MaxValue,
                    false,
                    false
                    );

                Func<IAccuracyCalculator> acfunc0 = () =>
                    new MetricsAccuracyCalculator(
                        new HalfSquaredEuclidianDistance(),
                        validation
                        );

                Classificate(validation, acfunc0);

            }

            if (File.Exists("grid.bmp"))
            {
                File.Delete("grid.bmp");
            }

            if (File.Exists("reconstruct.bmp"))
            {
                File.Delete("reconstruct.bmp");
            }

            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);

            {
                var validation = GetValidation(
                    int.MaxValue,
                    false,
                    false
                    );

                Func<IAccuracyCalculator> acfunc1 = () =>
                    new MetricsAccuracyCalculator2(
                        new HalfSquaredEuclidianDistance(),
                        validation
                        );

                Classificate(validation, acfunc1);
            }
        }

        private static void Classificate(
            IDataSet validation,
            Func<IAccuracyCalculator>  getAccuracy
            )
        {
            if (validation == null)
            {
                throw new ArgumentNullException("validation");
            }
            if (getAccuracy == null)
            {
                throw new ArgumentNullException("getAccuracy");
            }

            var randomizer = new DefaultRandomizer(983);

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));


            var mlp = mlpFactory.CreateMLP(
                "t",
                new IFunction[]
                {
                    null,
                    new RLUFunction(),
                    new RLUFunction(),
                    new LinearFunction(1f),
                },
                new int[]
                {
                    784,
                    500,
                    500,
                    784
                });

            using (var clProvider = new CLProvider())
            {
                var pcc = new CPUPropagatorComponentConstructor(
                    clProvider,
                    VectorizationSizeEnum.VectorizationMode16
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                pcc.CreateComponents(
                    mlp,
                    out containers,
                    out propagators
                    );

                var fp = new ForwardPropagation(
                    containers,
                    propagators,
                    mlp
                    );

                var c0 = getAccuracy();

                var drawer = new GridReconstructDrawer(
                    new MNISTVisualizerFactory(),
                    validation,
                    300,
                    new FileSystemArtifactContainer(
                        ".",
                        new SerializationHelper()));

                var before = DateTime.Now;

                IAccuracyRecord accuracyRecord;
                c0.CalculateAccuracy(
                    fp,
                    0,
                    drawer,
                    out accuracyRecord
                    );

                var after = DateTime.Now;

                Console.WriteLine("Error = " + DoubleConverter.ToExactString(accuracyRecord.PerItemError));

                Console.WriteLine(after - before);
            }
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
                        new ToAutoencoderDataItemTransformation(
                            dataItemFactory
                            );
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