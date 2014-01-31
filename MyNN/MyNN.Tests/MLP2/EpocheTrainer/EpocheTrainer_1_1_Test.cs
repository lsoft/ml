using System;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
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

            var mlp = new MLP(
                randomizer,
                ".",
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
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new CPUBackpropagationAlgorithm(
                                VectorizationSizeEnum.NoVectorization,
                                currentMLP,
                                currentConfig,
                                clProvider),
                        mlp,
                        validation,
                        config);

                alg.Train((epocheNumber) => dataset);
            }

        }
    }
}