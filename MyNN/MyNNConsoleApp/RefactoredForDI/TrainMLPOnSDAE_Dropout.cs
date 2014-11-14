using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MyNN;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Dense;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.Data.TypicalDataProvider;
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
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.GPU;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU;
using MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.CPU;
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
    public class TrainMLPOnSDAE_Dropout
    {
        public static void DoTrain()
        {
            var dataItemFactory = new DenseDataItemFactory();

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                100,//int.MaxValue,
                false,
                dataItemFactory
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                100,//int.MaxValue,
                false,
                dataItemFactory
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

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

            var filepath = "sdae20141108113834.tuned.sdae/epoche 248/sdae20141108094943.sdae";

            var mlp = serialization.LoadFromFile<MLP>(
                filepath
                );

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

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(true), false))
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


                //Func<int, INoiser> noiserProvider =
                //    (int epocheNumber) =>
                //    {
                //        if (epocheCount == epocheNumber)
                //        {
                //            return
                //                new NoNoiser();
                //        }

                //        var coef = (epocheCount - epocheNumber) / (float)epocheCount;
                //        //const float coef = 0.3f;

                //        var noiser = new ZeroMaskingNoiser(randomizer, coef * 0.25f, new RandomSeriesRange(randomizer, trainData[0].InputLength));

                //        //var noiser = new SequenceNoiser(
                //        //    randomizer,
                //        //    true,
                //        //    new GaussNoiser(coef * 0.20f, false, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                //        //    new MultiplierNoiser(randomizer, coef * 1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                //        //    new DistanceChangeNoiser(randomizer, coef * 1f, 3, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                //        //    new SaltAndPepperNoiser(randomizer, coef * 0.1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                //        //    new ZeroMaskingNoiser(randomizer, coef * 0.25f, new RandomSeriesRange(randomizer, trainData[0].InputLength))
                //        //    );

                //        return noiser;
                //    };

                //var noiserDataProvider = new NoiseDataProvider(
                //    trainData,
                //    noiserProvider,
                //    dataItemFactory
                //    );

                var noDeformationDataProvider = new NoDeformationTrainDataProvider(
                    trainData
                    );

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        //noiserDataProvider
                        noDeformationDataProvider
                        );

                var mlpContainer = rootContainer.GetChildContainer(mlpName);

                var mlpContainerHelper = new MLPContainerHelper();

                var maskContainerFactory = new BigArrayMaskContainerFactory(
                    randomizer,
                    clProvider
                    );

                const float p = 0.5f;

                var algo = new Backpropagation(
                    new CPUDropoutEpocheTrainer(
                        randomizer,
                        VectorizationSizeEnum.VectorizationMode16, 
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
                    config
                    );

                algo.Train(
                    trainDataProvider
                    );
            }
        }
    }
}
