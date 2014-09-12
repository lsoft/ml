using System;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    internal class EpocheTrainer_1_1_Test
    {
        public void ExecuteTest(
            DataSet dataset,
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
                    new ConstLearningRate(1f),
                    1,
                    0.0f,
                    1,
                    0.0f,
                    -1.0f);

                var validation = new EpocheTrainerValidation(
                    );

                var alg =
                    new BackpropagationAlgorithm(
                        new CPUBackpropagationEpocheTrainer(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            config,
                            clProvider),
                        new FileSystemArtifactContainer(".", new SerializationHelper()), //!!! переделать, нельзя использовать в тесте! 
                        mlp,
                        validation,
                        config);

                alg.Train(new NoDeformationTrainDataProvider(dataset));;
            }

        }
    }
}