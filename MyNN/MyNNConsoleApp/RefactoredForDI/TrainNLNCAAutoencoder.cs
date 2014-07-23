using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.KNN;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.ClassificationMLP.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainNLNCAAutoencoder
    {
        public static void DoTrain()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000//int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300//int.MaxValue
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

            var noiser = new AllNoisers(
                randomizer,
                new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                );

            var serialization = new SerializationHelper();

            var mlpf = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var firstLayerSize = trainData[0].Input.Length;
            const float lambda = 0.5f;
            const float partTakeOfAccount = 0.5f;

            var sa = new StackedAutoencoder(
                new IntelCPUDeviceChooser(),
                randomizer,
                mlpf,
                (IDataSet td) =>
                {
                    return
                        new NoiseDataProvider(
                            td,
                            noiser);
                },
                (IDataSet vd, IMLPContainer mlpContainer) =>
                {
                    var vda = vd.ConvertToAutoencoder();

                    return
                        new Validation(
                            new MetricsAccuracyCalculator(
                                new RMSE(),
                                vda),
                            new GridReconstructDrawer(
                                new MNISTVisualizer(), 
                                vda,
                                300,
                                100)
                            );
                },
                (int depthIndex) =>
                {
                    var lr =
                        depthIndex == 0
                            ? 0.005f
                            : 0.001f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        250,
                        0.0f,
                        25,
                        0f,
                        -0.0025f);

                    return conf;
                },
                new CPUNLNCABackpropagationAlgorithmFactory(
                    (data) =>
                        new DodfCalculatorOpenCL(
                            data,
                            new VectorizedCpuDistanceDictCalculator() //generation 1
                            ),
                    1,
                    lambda,
                    partTakeOfAccount),
                new CPUForwardPropagationFactory(),
                new LayerInfo(firstLayerSize, new RLUFunction()),
                new LayerInfo(400, new RLUFunction()),
                new LayerInfo(800, new RLUFunction())
                );

            var mlpName = string.Format(
                "nlnca_ae{0}.sdae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));


            var combinedNet = sa.Train(
                mlpName,
                new FileSystemMLPContainer(".", serialization),
                trainData,
                validationData
                );
        }
    }
}
