using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNNConsoleApp
{
    public class CalculateRBM
    {
        public static void Do(
            )
        {
            //var trainDataSetProvider = GetTrainProvider(
            //    10000,
            //    false,
            //    false
            //    );

            //var validationData = GetValidation(
            //    1000,
            //    false,
            //    false
            //    );

            var trainDataSetProvider = 
                new SmallDataSetProvider(
                    new SmallDataSet(
                    new List<IDataItem>
                    {
                        new DataItem(
                            new float[] { 0f, 1f, 0f },
                            new float[] { 1f }),
                    }
                    ));

            var validationData = new SmallDataSet(
                new List<IDataItem>
                {
                    new DataItem(
                        new float[] { 1f, 0f, 1f },
                        new float[] { 1f }),
                }
                );

            var serialization = new SerializationHelper();

            var rootContainer =
                new FileSystemArtifactContainer(
                    ".",
                    serialization);
                //new SavelessArtifactContainer(
                //    ".",
                //    serialization
                //    );

            var rbmContainer = rootContainer.GetChildContainer("RBM" + DateTime.Now.ToString("yyyyMMddHHmmss"));

            const int size = 2;

            var fec = new FloatArrayFreeEnergyCalculator(
                validationData.InputLength,
                size
                );

            var container = new FloatArrayContainer(
                new ConstRandomizer(0.5f, 5), 
                fec,
                validationData.InputLength,
                size
                );

            var calculator = new BBCalculator(
                new ConstRandomizer(0.5f, 5),
                validationData.InputLength,
                size
                );

            var algorithm = new CD(
                calculator,
                container
                );

            //var imager = new IsolatedImageReconstructor(
            //    validationData,
            //    100,
            //    28,
            //    28
            //    );

            //var featuree = new IsolatedFeatureExtractor(
            //    100,
            //    28,
            //    28
            //    );

            var rbm = new RBM(
                rbmContainer,
                container,
                algorithm,
                //imager,
                //featuree
                null,
                null
                );

            rbm.Train(
                trainDataSetProvider,
                validationData,
                new ConstLearningRate(0.1f),
                new AccuracyController(-0.1f, 1),
                1,
                1
                );
        }


        private static IDataSetProvider GetTrainProvider(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new CalculateBackprop.MNISTDataItemLoaderForDelete(
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

            var iteratorFactory = new DataIteratorFactory();

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation();
                    //new ToAutoencoderDataItemTransformation(
                    //    dataItemFactory);
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

            var dataItemLoader = new CalculateBackprop.MNISTDataItemLoaderForDelete(
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
                    //new ToAutoencoderDataItemTransformation(
                    //    dataItemFactory);
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
