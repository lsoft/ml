using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
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
    internal class EpocheTrainer_1_1_Test
    {
        public void ExecuteTest(
            IDataSet dataset,
            float weight0,
            float weight1,
            Func<IFunction> functionFactory)
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
                    1,
                    1
                });

            mlp.Layers[1].Neurons[0].Weights[0] = weight0;
            mlp.Layers[1].Neurons[0].Weights[1] = weight1;

            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new HalfSquaredEuclidianDistance(), 
                    new ConstLearningRate(1f),
                    1,
                    0.0f,
                    1,
                    0.0f,
                    -1.0f);

                var validation = new EpocheTrainerValidation(
                    );

                var mlpContainer = new MLPContainerHelper();

                var artifactContainer = new SavelessArtifactContainer(
                    ".",
                    new SerializationHelper()
                    );

                //var cc = new CPUPropagatorComponentConstructor(
                //    clProvider,
                //    VectorizationSizeEnum.NoVectorization
                //    );

                //ILayerContainer[] containers;
                //ILayerPropagator[] propagators;
                //cc.CreateComponents(
                //    mlp,
                //    out containers,
                //    out propagators);

                //var forwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                //    containers,
                //    propagators,
                //    mlp
                //    );

                var algof = new CPUBackpropagationFactory(
                    clProvider,
                    mlpContainer,
                    VectorizationSizeEnum.VectorizationMode16
                    );

                var algo = algof.CreateBackpropagation(
                    randomizer,
                    artifactContainer,
                    mlp,
                    validation,
                    config
                    );

                //var algo =
                //    new Backpropagation(
                //        new CPUEpocheTrainer(
                //            mlp,
                //            config,
                //            clProvider,
                //            containers.ToList().ConvertAll(j => j as IMemLayerContainer).ToArray(),
                //            forwardPropagation
                //            ),
                //        mlpContainer,
                //        artifactContainer,
                //        mlp,
                //        validation,
                //        config,
                //        forwardPropagation
                //        );

                var dataSetProvider = new TestDataSetProvider(
                    dataset
                    );

                algo.Train(dataSetProvider);
            }

        }
    }
}