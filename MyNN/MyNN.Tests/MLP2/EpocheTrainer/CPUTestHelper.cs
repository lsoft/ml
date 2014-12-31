using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    internal class CPUTestHelper
    {
        public IAccuracyRecord ExecuteTest(
            IDataSet dataset,
            Func<IFunction> functionFactory
            )
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }
            if (functionFactory == null)
            {
                throw new ArgumentNullException("functionFactory");
            }

            var randomizer = new ConstRandomizer(0.5f);

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));


            var mlpf = new MLPFactory(
                layerFactory
                );

            var mlp = mlpf.CreateMLP(
                DateTime.Now.ToString("yyyyMMddHHmmss"),
                new IFunction[]
                {
                    null,
                    functionFactory()
                },
                new int[]
                {
                    dataset.InputLength,
                    dataset.OutputLength
                });

            foreach (var l in mlp.Layers.Skip(1))
            {
                foreach (var n in l.Neurons)
                {
                    n.Weights.Fill(0.5f);
                    n.Bias = -0.5f;
                }
            }

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new ConstLearningRate(1 / 128f),
                10,
                0.0f,
                1,
                0.0f,
                -1.0f);

            var validation =
                new Validation(
                    new MetricsAccuracyCalculator(
                        new HalfSquaredEuclidianDistance(),
                        dataset),
                    null
                    );

            var mlpContainer = new MLPContainerHelper();

            var artifactContainer = new SavelessArtifactContainer(
                ".",
                new SerializationHelper()
                );

            using (var clProvider = new CLProvider())
            {
                var algof = new CPUBackpropagationFactory(
                    clProvider,
                    mlpContainer,
                    VectorizationSizeEnum.NoVectorization
                    );

                var algo = algof.CreateBackpropagation(
                    randomizer,
                    artifactContainer,
                    mlp,
                    validation,
                    config
                    );

                var dataSetProvider = new TestDataSetProvider(
                    dataset
                    );

                var acc = algo.Train(dataSetProvider);

                return acc;
            }
        }

    }
}