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
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Backpropagator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Backpropagator;
using MyNN.MLP.Classic.ForwardPropagation.CSharp;
using MyNN.MLP.Convolution.Activator;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.Connector;
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
    public class Convolute4
    {
        const int ImageSize = 28;


        const int KernelSize0 = 5;
        const int ConvolutionSize0 = ImageSize - KernelSize0 + 1;
        const float ScaleFactor0 = 0.5f;
        const int FeatureMapCount0 = 5;

        const int KernelSize1 = 3;
        const int ConvolutionSize1 = (int)(ConvolutionSize0 * ScaleFactor0) - KernelSize1 + 1;
        const float ScaleFactor1 = 0.5f;
        const int FeatureMapCount1 = 12;

        const int EpochCount = 50;
        const float LearningRate = 0.001f;
        const int VizCount = 100;

        public static void Do(
            )
        {

            var trainDataSetProvider = GetTrainProvider(
                60000,
                true,
                false
                );

            var validationData = GetValidation(
                10000,
                true,
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

            var l1Dimension = new Dimension(2, ConvolutionSize0, ConvolutionSize0);
            var l1KernelDimension = new Dimension(2, KernelSize0, KernelSize0);
            var l1 = new ConvolutionLayer(
                neuronFactory,
                //new SigmoidFunction(1f), 
                new HyperbolicTangensFunction(), 
                //new RLUFunction(),
                l1Dimension,
                FeatureMapCount0,
                l1KernelDimension,
                //new ConvolutionWeightBiasIniter(randomizer, l1KernelDimension, FeatureMapCount0)
                new RandomWeightBiasIniter(randomizer)
                );

            var l2 = new AvgPoolingLayer(
                neuronFactory,
                new Dimension(2, (int) (ConvolutionSize0*ScaleFactor0), (int) (ConvolutionSize0*ScaleFactor0)),
                FeatureMapCount0,
                ScaleFactor0
                );

            var l3Dimension = new Dimension(2, ConvolutionSize1, ConvolutionSize1);
            var l3KernelDimension = new Dimension(2, KernelSize1, KernelSize1);
            var l3 = new ConvolutionLayer(
                neuronFactory,
                //new SigmoidFunction(1f), 
                new HyperbolicTangensFunction(), 
                //new RLUFunction(),
                l3Dimension,
                FeatureMapCount1,
                l3KernelDimension,
                //new ConvolutionWeightBiasIniter(randomizer, l3KernelDimension, FeatureMapCount1)
                new RandomWeightBiasIniter(randomizer)
                );

            var l4 = new AvgPoolingLayer(
                neuronFactory,
                new Dimension(2, (int)(ConvolutionSize1 * ScaleFactor1), (int)(ConvolutionSize1 * ScaleFactor1)),
                FeatureMapCount1,
                ScaleFactor1
                );

            var l5 = new FullConnectedLayer(
                neuronFactory,
                //new SigmoidFunction(1f),
                new HyperbolicTangensFunction(),
                new Dimension(1, 150),
                l4.TotalNeuronCount
                );

            var l6 = new FullConnectedLayer(
                neuronFactory,
                new SigmoidFunction(1f),
                new Dimension(1, 10),
                l5.TotalNeuronCount
                );

            var mlp = mlpFactory.CreateMLP(
                "conv" + DateTime.Now.ToString("yyyyMMddHHmmss"),
                new ILayer[]
                {
                    l0,
                    l1,
                    l2,
                    l3,
                    l4,
                    l5,
                    l6
                }
                );

            var convolutionCalculator = new NaiveConvolutionCalculator();

            var functionActivator = new FunctionActivator();

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new LinearLearningRate(LearningRate, 0.99f), 
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
                new FileSystemArtifactContainer(
                    ".",
                    serialization);
                //new SavelessArtifactContainer(
                //    ".",
                //    serialization
                //    );

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

            //var connector = new FullConnectedConnector(
            //    (mlp.Layers[2] as IAvgPoolingLayer).GetConfiguration()
            //    );

            var connector = new NaiveConnector(
                (mlp.Layers[2] as IAvgPoolingLayer).GetConfiguration(),
                3
                );
            
            //----------------------------------------------------------------------

            var containers = new ILayerContainer[mlp.Layers.Length];

            containers[0] = new CSharpLayerContainer(
                mlp.Layers[0].GetConfiguration()
                );
            containers[1] = new CSharpConvolutionLayerContainer(
                mlp.Layers[1].GetConfiguration() as IConvolutionLayerConfiguration
                );
            containers[2] = new CSharpAvgPoolingLayerContainer(
                mlp.Layers[2].GetConfiguration() as IAvgPoolingLayerConfiguration
                );
            containers[3] = new CSharpConvolutionLayerContainer(
                mlp.Layers[3].GetConfiguration() as IConvolutionLayerConfiguration
                );
            containers[4] = new CSharpAvgPoolingLayerContainer(
                mlp.Layers[4].GetConfiguration() as IAvgPoolingLayerConfiguration
                );
            containers[5] = new CSharpLayerContainer(
                mlp.Layers[5].GetConfiguration()
                );
            containers[6] = new CSharpLayerContainer(
                mlp.Layers[6].GetConfiguration()
                );

            //----------------------------------------------------------------------

            var propagators = new ILayerPropagator[mlp.Layers.Length];
            propagators[1] = new CSharpFullConnected_ConvolutionLayerPropagator(
                containers[0] as ICSharpLayerContainer,
                containers[1] as ICSharpConvolutionLayerContainer,
                convolutionCalculator,
                functionActivator
                );
            propagators[2] = new CSharpConvolution_AvgPoolingLayerPropagator(
                containers[1] as ICSharpConvolutionLayerContainer,
                containers[2] as ICSharpAvgPoolingLayerContainer
                );
            propagators[3] = new CSharpAvgPooling_ConvolutionLayerPropagator(
                containers[2] as ICSharpAvgPoolingLayerContainer,
                containers[3] as ICSharpConvolutionLayerContainer,
                convolutionCalculator,
                functionActivator,
                connector
                );
            propagators[4] = new CSharpConvolution_AvgPoolingLayerPropagator(
                containers[3] as ICSharpConvolutionLayerContainer,
                containers[4] as ICSharpAvgPoolingLayerContainer
                );
            propagators[5] = new CSharpLayerPropagator(
                mlp.Layers[5],
                containers[4] as ICSharpLayerContainer,
                containers[5] as ICSharpLayerContainer
                );
            propagators[6] = new CSharpLayerPropagator(
                mlp.Layers[6],
                containers[5] as ICSharpLayerContainer,
                containers[6] as ICSharpLayerContainer
                );

            //----------------------------------------------------------------------

            //dedy aggregators

            ICSharpDeDyAggregator dedyAggregator0 = null;

            ICSharpDeDyAggregator dedyAggregator1 = new CSharpStubConvolutionDeDyAggregator(
                (mlp.Layers[1] as IConvolutionLayer).GetConfiguration()
                );

            ICSharpDeDyAggregator dedyAggregator2 = new CSharpAvgPoolingDeDyAggregator(
                (mlp.Layers[1] as IConvolutionLayer).GetConfiguration(),
                (mlp.Layers[2] as IAvgPoolingLayer).GetConfiguration()
                );

            ICSharpDeDyAggregator dedyAggregator3 = new CSharpConvolutionDeDyAggregator(
                (mlp.Layers[2] as IAvgPoolingLayer).GetConfiguration(),
                containers[3] as ICSharpConvolutionLayerContainer, 
                convolutionCalculator,
                connector
                );

            ICSharpDeDyAggregator dedyAggregator4 = new CSharpAvgPoolingDeDyAggregator(
                (mlp.Layers[3] as IConvolutionLayer).GetConfiguration(),
                (mlp.Layers[4] as IAvgPoolingLayer).GetConfiguration()
                );

            var dedyAggregator5 = new CSharpDeDyAggregator(
                mlp.Layers[4].TotalNeuronCount,
                mlp.Layers[5].TotalNeuronCount,
                (containers[5] as ICSharpLayerContainer).WeightMem
                );

            var dedyAggregator6 = new CSharpDeDyAggregator(
                mlp.Layers[5].TotalNeuronCount,
                mlp.Layers[6].TotalNeuronCount,
                (containers[6] as ICSharpLayerContainer).WeightMem
                );

            //----------------------------------------------------------------------

            var fp = new ForwardPropagation(
                containers,
                propagators,
                mlp
                );


            var backpropagator6 =
                new CSharpOutputLayerBackpropagator(
                    config,
                    containers[5] as ICSharpLayerContainer,
                    containers[6] as ICSharpLayerContainer,
                    desiredValuesContainer,
                    dedyAggregator6
                    );

            var backpropagator5 =
                new CSharpHiddenLayerBackpropagator(
                    config,
                    true,
                    containers[4] as ICSharpLayerContainer,
                    containers[5] as ICSharpLayerContainer,
                    dedyAggregator6,
                    dedyAggregator5
                    );

            var backpropagator4 = new CSharpAvgPoolingFullConnectedBackpropagator(
                containers[4] as ICSharpAvgPoolingLayerContainer,
                dedyAggregator5,
                dedyAggregator4
                );

            var backpropagator3 =
                new CSharpConvolutionPoolingLayerBackpropagator(
                    config,
                    (mlp.Layers[4] as IAvgPoolingLayer).GetConfiguration(),
                    containers[2] as ICSharpLayerContainer,
                    containers[3] as ICSharpConvolutionLayerContainer,
                    dedyAggregator4,
                    dedyAggregator3,
                    true
                    );

            var backpropagator2 = new CSharpAvgPoolingConvolutionBackpropagator(
                (containers[2] as ICSharpAvgPoolingLayerContainer).Configuration,
                dedyAggregator3,
                dedyAggregator2
                );

            var backpropagator1 =
                new CSharpConvolutionPoolingLayerBackpropagator(
                    config,
                    (mlp.Layers[2] as IAvgPoolingLayer).GetConfiguration(),
                    containers[0] as ICSharpLayerContainer,
                    containers[1] as ICSharpConvolutionLayerContainer,
                    dedyAggregator2,
                    dedyAggregator1,
                    false
                    );

            var backpropagators = new ILayerBackpropagator[]
            {
                null,
                backpropagator1, 
                backpropagator2,
                backpropagator3,
                backpropagator4,
                backpropagator5,
                backpropagator6
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
                    weights.GetSubArray(fmi * KernelSize0 * KernelSize0, KernelSize0 * KernelSize0),
                    KernelSize0,
                    KernelSize0
                    );

                var kernelBiasContainer = new ReferencedKernelBiasContainer(
                    new Dimension(2, KernelSize0, KernelSize0),
                    weights,
                    fmi * KernelSize0 * KernelSize0,
                    biases,
                    fmi
                    );

                var count = 0;
                using(var iter = validationData.StartIterate())
                while(iter.MoveNext() && count++ < VizCount)
                {
                    var vi = iter.Current;

                    var net = new ReferencedSquareFloat(
                        new Dimension(2, ConvolutionSize0, ConvolutionSize0),
                        (containers[1] as ICSharpLayerContainer).NetMem,
                        fmi * ConvolutionSize0 * ConvolutionSize0
                        );

                    var state = new ReferencedSquareFloat(
                        new Dimension(2, ConvolutionSize0, ConvolutionSize0),
                        (containers[1] as ICSharpLayerContainer).StateMem,
                        fmi * ConvolutionSize0 * ConvolutionSize0
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
                            fmi * ConvolutionSize0 * ConvolutionSize0,
                            ConvolutionSize0 * ConvolutionSize0)
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
            for (var y = 0; y < ConvolutionSize0; y++)
            {
                for (var x = 0; x < ConvolutionSize0; x++)
                {
                    result[((ImageSize - ConvolutionSize0) + y)*ImageSize + (ImageSize - ConvolutionSize0) + x] = f[i];

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
