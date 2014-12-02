using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2.Forward
{
    internal class ForwardStateTester
    {
        public Pair<Pair<float, float>, Pair<float, float>> ExecuteTestWith_1_2_2_MLP(
            IDataSet dataset,
            List<float> hiddenWeights,
            List<float> outputWeights,
            Func<IFunction> functionFactory,
            Func<IMLP, IForwardPropagation> forwardFactory)
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }
            if (hiddenWeights == null)
            {
                throw new ArgumentNullException("hiddenWeights");
            }
            if (outputWeights == null)
            {
                throw new ArgumentNullException("outputWeights");
            }
            if (functionFactory == null)
            {
                throw new ArgumentNullException("functionFactory");
            }
            if (forwardFactory == null)
            {
                throw new ArgumentNullException("forwardFactory");
            }
            if (hiddenWeights.Count != 4)
            {
                throw new ArgumentException("hiddenWeights.Count != 4");
            }
            if (outputWeights.Count != 6)
            {
                throw new ArgumentException("outputWeights.Count != 6");
            }

            var randomizer = new ConstRandomizer(0.5f);

            var mlpf = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var mlp = mlpf.CreateMLP(
                DateTime.Now.ToString("yyyyMMddHHmmss"),
                new IFunction[]
                {
                    null,
                    functionFactory(),
                    functionFactory()
                },
                new int[]
                {
                    1,
                    2,
                    2
                });

            mlp.Layers[1].Neurons[0].Weights[0] = hiddenWeights[0];
            mlp.Layers[1].Neurons[0].Weights[1] = hiddenWeights[1];
            mlp.Layers[1].Neurons[1].Weights[0] = hiddenWeights[1];
            mlp.Layers[1].Neurons[1].Weights[1] = hiddenWeights[2];

            mlp.Layers[2].Neurons[0].Weights[0] = outputWeights[0];
            mlp.Layers[2].Neurons[0].Weights[1] = outputWeights[1];
            mlp.Layers[2].Neurons[0].Weights[2] = outputWeights[2];
            mlp.Layers[2].Neurons[1].Weights[0] = outputWeights[3];
            mlp.Layers[2].Neurons[1].Weights[1] = outputWeights[4];
            mlp.Layers[2].Neurons[1].Weights[2] = outputWeights[5];

            var forward = forwardFactory(mlp);

            var output = forward.ComputeState(dataset);

            Assert.IsTrue(output.Count == 1);
            Assert.IsTrue(output[0].LState.Length == 3);

            return
                new Pair<Pair<float, float>, Pair<float, float>>(
                    new Pair<float, float>(
                        output[0].LState[1].NState[0],
                        output[0].LState[1].NState[1]
                        ),
                    new Pair<float, float>(
                        output[0].LState[2].NState[0],
                        output[0].LState[2].NState[1]
                        )
                );
        }

        public Pair<float, float> ExecuteTestWith_1_1_1_MLP(
            IDataSet dataset,
            List<float> hiddenWeights,
            List<float> outputWeights, 
            Func<IFunction> functionFactory,
            Func<IMLP, IForwardPropagation> forwardFactory)
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }
            if (hiddenWeights == null)
            {
                throw new ArgumentNullException("hiddenWeights");
            }
            if (outputWeights == null)
            {
                throw new ArgumentNullException("outputWeights");
            }
            if (functionFactory == null)
            {
                throw new ArgumentNullException("functionFactory");
            }
            if (forwardFactory == null)
            {
                throw new ArgumentNullException("forwardFactory");
            }
            if (hiddenWeights.Count != 2)
            {
                throw new ArgumentException("hiddenWeights.Count != 2");
            }
            if (outputWeights.Count != 2)
            {
                throw new ArgumentException("outputWeights.Count != 2");
            }

            var randomizer = new ConstRandomizer(0.5f);

            var mlpf = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var mlp = mlpf.CreateMLP(
                DateTime.Now.ToString("yyyyMMddHHmmss"),
                new IFunction[]
                {
                    null,
                    functionFactory(),
                    functionFactory()
                },
                new int[]
                {
                    1,
                    1,
                    1
                });

            mlp.Layers[1].Neurons[0].Weights[0] = hiddenWeights[0];
            mlp.Layers[1].Neurons[0].Weights[1] = hiddenWeights[1];

            mlp.Layers[2].Neurons[0].Weights[0] = outputWeights[0];
            mlp.Layers[2].Neurons[0].Weights[1] = outputWeights[1];

            var forward = forwardFactory(mlp);

            var output = forward.ComputeState(dataset);

            Assert.IsTrue(output.Count == 1);
            Assert.IsTrue(output[0].LState.Length == 3);

            return
                new Pair<float, float>(
                    output[0].LState[1].NState[0],
                    output[0].LState[2].NState[0]
                    );
        }

    }
}
