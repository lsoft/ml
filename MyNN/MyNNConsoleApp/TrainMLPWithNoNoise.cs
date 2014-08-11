using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp
{
    public class TrainMLPWithNoNoise
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

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            using (var clProvider = new CLProvider())
            {
                var mlpName = string.Format(
                    "justmlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var validation = new Validation(
                    new FeatureAndMetricsAccuracyCalculator(
                        mlpName,
                        clProvider,
                        new HalfSquaredEuclidianDistance(),
                        validationData),
                    new GridReconstructDrawer(
                        new MNISTVisualizer(),
                        validationData,
                        300,
                        100)
                    );

                var mlp = mlpfactory.CreateMLP(
                    mlpName,
                    new IFunction[]
                    {
                        null,
                        new RLUFunction(), 
                        new RLUFunction(), 
                    },
                    new int[]
                    {
                        784,
                        1000,
                        784
                    });

                var config = new LearningAlgorithmConfig(
                    new ConstLearningRate(0.0001f), 
                    1,
                    0f,
                    100,
                    -1f,
                    -1f
                    );

                var storedEpochNumber = 0;

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoiseDataProvider(
                            trainData,
                            epochNumber =>
                            {
                                if (epochNumber < 50)
                                {
                                    var noiser = new AllNoisers(
                                        randomizer,
                                        new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                                        new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                                        new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                                        new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                                        new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                                        );

                                    if (storedEpochNumber != epochNumber)
                                    {
                                        storedEpochNumber = epochNumber;

                                        ConsoleAmbientContext.Console.WriteLine(
                                            string.Format(
                                                "--- epoch {0} --- FULL NOISE",
                                                epochNumber
                                                ));
                                    }

                                    return
                                        noiser;
                                }
                                else if (epochNumber < 70)
                                {
                                    var coef = (71 - epochNumber)/21f;

                                    var noiser = new AllNoisers(
                                        randomizer,
                                        new GaussNoiser(coef * 0.20f, false, new RandomRange(randomizer)),
                                        new MultiplierNoiser(randomizer, coef * 1f, new RandomRange(randomizer)),
                                        new DistanceChangeNoiser(randomizer, coef * 1f, 3, new RandomRange(randomizer)),
                                        new SaltAndPepperNoiser(randomizer, coef * 0.1f, new RandomRange(randomizer)),
                                        new ZeroMaskingNoiser(randomizer, coef * 0.25f, new RandomRange(randomizer))
                                        );

                                    if (storedEpochNumber != epochNumber)
                                    {
                                        storedEpochNumber = epochNumber;
                                        
                                        ConsoleAmbientContext.Console.WriteLine(
                                            string.Format(
                                                "--- epoch {0} --- {1} NOISE",
                                                epochNumber,
                                                coef
                                                ));
                                    }

                                    return
                                        noiser;
                                }
                                else
                                {
                                    if (storedEpochNumber != epochNumber)
                                    {
                                        storedEpochNumber = epochNumber;

                                        ConsoleAmbientContext.Console.WriteLine(
                                            string.Format(
                                                "--- epoch {0} --- NO NOISE",
                                                epochNumber
                                                ));
                                    }

                                    return
                                        null;
                                }
                            })
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

    public class FeatureAndMetricsAccuracyCalculator : IAccuracyCalculator
    {
        private readonly string _mlpName;
        private readonly CLProvider _clProvider;
        private readonly IMetrics _metrics;
        private readonly IDataSet _validationData;

        public FeatureAndMetricsAccuracyCalculator(
            string mlpName,
            CLProvider clProvider,
            IMetrics metrics,
            IDataSet validationData
            )
        {
            if (mlpName == null)
            {
                throw new ArgumentNullException("mlpName");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (metrics == null)
            {
                throw new ArgumentNullException("metrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _mlpName = mlpName;
            _clProvider = clProvider;
            _metrics = metrics;
            _validationData = validationData;
        }

        public void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            out List<ILayerState> netResults,
            out IAccuracyRecord accuracyRecord
            )
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            var fff = new FileSystemFeatureVisualization(
                new NoRandomRandomizer(),
                new SerializationHelper().DeepClone(forwardPropagation.MLP),
                new ForwardPropagationFactory(
                    new CPUPropagatorComponentConstructor(
                        _clProvider,
                        VectorizationSizeEnum.VectorizationMode16)));

            fff.Visualize(
                new MNISTVisualizer(),
                string.Format("{1}/_{0}_feature.bmp", epocheNumber != null ? epocheNumber.Value : -1, _mlpName),
                25,
                5f,
                900,
                false,
                true);

            netResults = forwardPropagation.ComputeOutput(_validationData);

            //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
            //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
            var d = new List<DataItem>(_validationData.Count + 1);
            for (var i = 0; i < _validationData.Count; i++)
            {
                d.Add(
                    new DataItem(
                        _validationData[i].Output,
                        netResults[i].NState));
            }

            var totalError = d.AsParallel().Sum(
                j => _metrics.Calculate(j.Input, j.Output));

            var perItemError = totalError / _validationData.Count;

            accuracyRecord = new MetricAccuracyRecord(
                epocheNumber ?? 0,
                perItemError);
        }

    }

}
