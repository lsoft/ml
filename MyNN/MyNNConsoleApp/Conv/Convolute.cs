using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
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
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Classic.ForwardPropagation.CSharp;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ErrorCalculator;
using MyNN.MLP.Convolution.ErrorCalculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNNConsoleApp.Conv
{
    public class Convolute
    {
        const int ImageSize = 6;
        const int KernelSize = 3;
        const int ConvolutionSize = ImageSize - KernelSize + 1;
        const int EpochCount = 150;

        //const int ImageSize = 28;
        //const int KernelSize = 5;
        //const int ConvolutionSize = ImageSize - KernelSize + 1;
        //const int EpochCount = 150;
        
        public static void Do(
            )
        {
            var validationData = GetMiniDataSet(
                );

            var randomizer =
                new DefaultRandomizer(1);

            var neuronFactory = new NeuronFactory(randomizer);

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    neuronFactory
                    )
                );

            var l0 = new FullConnectedLayer(
                neuronFactory,
                new Dimension(2, ImageSize, ImageSize)
                );

            var l1 = new ConvolutionLayer(
                randomizer,
                neuronFactory,
                new SigmoidFunction(
                    1f), 
                new Dimension(2, ConvolutionSize, ConvolutionSize),
                new Dimension(2, KernelSize, KernelSize)
                );


            var mlp = mlpFactory.CreateMLP(
                "conv" + DateTime.Now.ToString("yyyyMMddHHmmss"),
                new ILayer[]
                {
                    l0,
                    l1
                }
                );

            var convolutionCalculator = new NaiveConvolutionCalculator();

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new ConstLearningRate(0.5f), 
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

            var validation = new ConvolutionValidation(
                validationData
                );

            var errorCalculator = new NaiveErrorCalculator();

            //----------------------------------------------------------------------

            var containers = new ILayerContainer[mlp.Layers.Length];

            containers[0] = new CSharpLayerContainer(
                mlp.Layers[0].GetConfiguration()
                );
            containers[1] = new CSharpLayerContainer(
                mlp.Layers[1].GetConfiguration()
                );

            var propagators = new ILayerPropagator[mlp.Layers.Length];
            propagators[1] = new CSharpConvolutionLayerPropagator(
                mlp.Layers[0],
                mlp.Layers[1],
                containers[0] as ICSharpLayerContainer,
                containers[1] as ICSharpLayerContainer,
                convolutionCalculator
                );

            var fp = new ForwardPropagation(
                containers,
                propagators,
                mlp
                );


            var backpropagator1 =
                new CSharpConvolutionOutputLayerBackpropagator(
                    mlp,
                    config,
                    containers[0] as ICSharpLayerContainer,
                    containers[1] as ICSharpLayerContainer,
                    desiredValuesContainer,
                    errorCalculator
                    );

            var backpropagators = new ILayerBackpropagator[]
            {
                null,
                backpropagator1
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
                new SmallDataSetProvider(validationData)
                );

        }


        #region mini dataset

        private static IDataSet GetMiniDataSet(
            )
        {
            var imagea = new float[ImageSize*ImageSize];
            var image = new ReferencedSquareFloat(
                new Dimension(2, ImageSize, ImageSize),
                imagea,
                0);
            for (var cc = 0; cc < ImageSize; cc++)
            {
                image.SetValueFromCoordSafely(ImageSize - cc - 1, cc, 1f);
            }

            var desiredValuesa = new float[ConvolutionSize*ConvolutionSize];
            desiredValuesa.Fill(0.15f);
            var desiredValues = new ReferencedSquareFloat(
                new Dimension(2, ConvolutionSize, ConvolutionSize),
                desiredValuesa,
                0);
            for (var cc = 0; cc < ConvolutionSize; cc++)
            {
                desiredValues.SetValueFromCoordSafely(ConvolutionSize - cc - 1, cc, 0.85f);
            }

            var ds = new SmallDataSet(
                new List<IDataItem>
                {
                    new DataItem(
                        imagea,
                        desiredValuesa
                        )
                });

            return ds;
        }

        private class ConvolutionValidation : IValidation
        {
            private readonly IDataSet _validationData;

            public ConvolutionValidation(
                IDataSet validationData
                )
            {
                if (validationData == null)
                {
                    throw new ArgumentNullException("validationData");
                }

                _validationData = validationData;
            }

            public IAccuracyRecord Validate(
                IForwardPropagation forwardPropagation,
                int? epocheNumber,
                IArtifactContainer epocheContainer
                )
            {
                var output = forwardPropagation.ComputeOutput(_validationData);

                using(var i = _validationData.StartIterate())
                {
                    i.MoveNext();

                    LayerVisualizer.Show(
                        "desired",
                        i.Current.Output,
                        ConvolutionSize,
                        ConvolutionSize
                    );
                }

                LayerVisualizer.Show(
                    "result",
                    output[0].NState,
                    ConvolutionSize,
                    ConvolutionSize
                    );

                Console.ReadLine();

                return 
                    new MetricAccuracyRecord(epocheNumber != null ? epocheNumber.Value : 0, 1f);

            }
        }

        #endregion
    }
}
