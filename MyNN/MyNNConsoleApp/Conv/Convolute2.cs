using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.IterateHelper;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.MNIST;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Backpropagator;
using MyNN.MLP.Classic.ForwardPropagation.CSharp;
using MyNN.MLP.Convolution.Activator;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Layer.WeightBiasIniter;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNNConsoleApp.Conv
{
    public class Convolute2
    {
        const int ImageSize = 28;
        const int KernelSize = 5;
        const int ConvolutionSize = ImageSize - KernelSize + 1;
        const int EpochCount = 5;
        const float LearningRate = 0.01f;
        const int VizCount = 100;
        private const int FeatureMapCount = 5;

        public static void Do(
            )
        {

            var trainDataSetProvider = GetTrainProvider(
                6000,
                false,
                false
                );

            var validationData = GetValidation(
                1000,
                false,
                false
                );

            var randomizer =
                new DefaultRandomizer(2);

            var neuronFactory = new NeuronFactory(randomizer);

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    neuronFactory));

            var l0 = new FullConnectedLayer(
                neuronFactory,
                new Dimension(2, ImageSize, ImageSize)
                );

            var l1Dimension = new Dimension(2, ConvolutionSize, ConvolutionSize);
            var l1KernelDimension = new Dimension(2, KernelSize, KernelSize);
            var l1 = new ConvolutionLayer(
                neuronFactory,
                new RLUFunction(),
                l1Dimension,
                FeatureMapCount,
                l1KernelDimension,
                new RandomWeightBiasIniter(randomizer)
                );

            var l2 = new FullConnectedLayer(
                neuronFactory,
                new SigmoidFunction(1f),
                new Dimension(1, 10),
                l1.TotalNeuronCount
                );

            var mlp = mlpFactory.CreateMLP(
                "conv" + DateTime.Now.ToString("yyyyMMddHHmmss"),
                new ILayer[]
                {
                    l0,
                    l1, 
                    l2
                }
                );

            var convolutionCalculator = new NaiveConvolutionCalculator();

            var functionActivator = new FunctionActivator();

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new ConstLearningRate(LearningRate), 
                1,
                0f,
                EpochCount,
                0.0001f
                );

            var desiredValuesContainer = new CSharpDesiredValuesContainer(
                mlp
                );

            var serialization = new SerializationHelper();

            var rootContainer = 
                //new FileSystemArtifactContainer(
                //    ".",
                //    serialization);
                new SavelessArtifactContainer(
                    ".",
                    serialization
                    );

            var mlpName = string.Format("conv{0}.mlp", DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlpContainer = rootContainer.GetChildContainer(mlpName);

            var mlpContainerHelper = new MLPContainerHelper();

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                null
                );

            //----------------------------------------------------------------------

            var containers = new ILayerContainer[mlp.Layers.Length];

            containers[0] = new CSharpLayerContainer(
                mlp.Layers[0].GetConfiguration()
                );
            containers[1] = new CSharpConvolutionLayerContainer(
                mlp.Layers[1].GetConfiguration() as IConvolutionLayerConfiguration
                );
            containers[2] = new CSharpLayerContainer(
                mlp.Layers[2].GetConfiguration()
                );

            //----------------------------------------------------------------------

            var propagators = new ILayerPropagator[mlp.Layers.Length];
            propagators[1] = new CSharpFullConnected_ConvolutionLayerPropagator(
                containers[0] as ICSharpLayerContainer,
                containers[1] as ICSharpConvolutionLayerContainer,
                convolutionCalculator,
                functionActivator
                );
            propagators[2] = new CSharpLayerPropagator(
                mlp.Layers[2],
                containers[1] as ICSharpLayerContainer,
                containers[2] as ICSharpLayerContainer
                );

            //----------------------------------------------------------------------

            ICSharpDeDyAggregator dedyAggregator0 = null;

            ICSharpDeDyAggregator dedyAggregator1 = null;

            var dedyAggregator2 = new CSharpDeDyAggregator(
                mlp.Layers[1].TotalNeuronCount,
                mlp.Layers[2].TotalNeuronCount,
                (containers[2] as ICSharpLayerContainer).WeightMem
                );

            //----------------------------------------------------------------------

            var fp = new ForwardPropagation(
                containers,
                propagators,
                mlp
                );

            //----------------------------------------------------------------------
            
            var backpropagator2 =
                new CSharpOutputLayerBackpropagator(
                    config,
                    containers[1] as ICSharpLayerContainer,
                    containers[2] as ICSharpLayerContainer,
                    desiredValuesContainer,
                    dedyAggregator2
                    );

            var backpropagator1 =
                new CSharpConvolutionFullConnectedLayerBackpropagator(
                    config,
                    containers[0] as ICSharpLayerContainer,
                    containers[1] as ICSharpConvolutionLayerContainer,
                    dedyAggregator2
                    );

            var backpropagators = new ILayerBackpropagator[]
            {
                null,
                backpropagator1, 
                backpropagator2
            };

            var bp = new Backpropagation(
                new EpocheTrainer(
                    mlp,
                    config,
                    containers,
                    desiredValuesContainer,
                    backpropagators,
                    () => { },
                    fp
                    ),
                mlpContainerHelper,
                mlpContainer,
                mlp,
                validation,
                config,
                fp
                );

            bp.Train(
                trainDataSetProvider
                );

            //----------------------------------------------------------------------

            float[] weights;
            float[] biases;
            l1.GetClonedWeights(
               out weights,
               out biases
               );

            for (var fmi = 0; fmi < l1.FeatureMapCount; fmi++)
            {
                var viz = new MNISTVisualizer(VizCount);

                LayerVisualizer.Show(
                    "kernel " + fmi,
                    weights.GetSubArray(fmi * KernelSize * KernelSize, KernelSize * KernelSize),
                    KernelSize,
                    KernelSize
                    );

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    new Dimension(2, KernelSize, KernelSize),
                    weights,
                    fmi * KernelSize * KernelSize,
                    biases,
                    fmi
                    );

                var count = 0;
                using(var iter = validationData.StartIterate())
                while(iter.MoveNext() && count++ < VizCount)
                {
                    var vi = iter.Current;

                    var net = new ReferencedSquareFloat(
                        new Dimension(2, ConvolutionSize, ConvolutionSize),
                        (containers[1] as ICSharpLayerContainer).NetMem,
                        fmi * ConvolutionSize * ConvolutionSize
                        );

                    var state = new ReferencedSquareFloat(
                        new Dimension(2, ConvolutionSize, ConvolutionSize),
                        (containers[1] as ICSharpLayerContainer).StateMem,
                        fmi * ConvolutionSize * ConvolutionSize
                        );

                    convolutionCalculator.CalculateConvolutionWithOverwrite(
                        kernelBiasContainer,
                        new ReferencedSquareFloat(new Dimension(2, 28, 28), vi.Input, 0),
                        net
                        );

                    functionActivator.Apply(
                        l1.LayerActivationFunction,
                        net,
                        state
                        );

                    var rescaled = Rescale(
                        (containers[1] as ICSharpLayerContainer).StateMem.GetSubArray(
                            fmi * ConvolutionSize * ConvolutionSize,
                            ConvolutionSize * ConvolutionSize)
                        );

                    var p = new Pair<float[], float[]>(
                        vi.Input,
                        rescaled
                        );

                    viz.VisualizePair(p);
                }

                using (var fs = new FileStream(string.Format("_p{0}.bmp", fmi), FileMode.Create, FileAccess.Write))
                {
                    viz.SavePairs(fs);
                }
            }
        }


        private static  float[] Rescale(
            float[] f
            )
        {
            var result = new float[ImageSize*ImageSize];

            var i = 0;
            for (var y = 0; y < ConvolutionSize; y++)
            {
                for (var x = 0; x < ConvolutionSize; x++)
                {
                    result[((ImageSize - ConvolutionSize) + y)*ImageSize + (ImageSize - ConvolutionSize) + x] = f[i];

                    i++;
                }
            }

            return result;
        }

        private static IDataSetProvider GetTrainProvider(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
        )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoaderForDelete(
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

            var dataItemLoader = new MNISTDataItemLoaderForDelete(
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
