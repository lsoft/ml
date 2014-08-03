﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward
{
    internal class ForwardTester
    {

        public float ExecuteTestWith_1_1_MLP(
            IDataSet dataset,
            float weight0,
            float weight1,
            Func<IFunction> functionFactory,
            Func<IMLP, IForwardPropagation> forwardFactory)
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }
            if (functionFactory == null)
            {
                throw new ArgumentNullException("functionFactory");
            }
            if (forwardFactory == null)
            {
                throw new ArgumentNullException("forwardFactory");
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
                    functionFactory()
                },
                new int[]
                {
                    1,
                    1
                });

            mlp.Layers[1].Neurons[0].Weights[0] = weight0;
            mlp.Layers[1].Neurons[0].Weights[1] = weight1;
            
            var forward = forwardFactory(mlp);

            var output = forward.ComputeOutput(dataset);

            Assert.IsTrue(output.Count == 1);
            Assert.IsTrue(output[0].State.Length == 1);

            return
                output[0].State[0];
        }

        public float ExecuteTestWith_5_24_24_1_MLP(
            IDataSet dataset,
            Func<IFunction> functionFactory,
            Func<IMLP, IForwardPropagation> forwardFactory)
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }
            if (functionFactory == null)
            {
                throw new ArgumentNullException("functionFactory");
            }
            if (forwardFactory == null)
            {
                throw new ArgumentNullException("forwardFactory");
            }

            var randomizer = new ConstRandomizer(0.125f);

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
                    functionFactory(),
                    functionFactory()
                },
                new int[]
                {
                    5,
                    24,
                    24,
                    1
                });

            var forward = forwardFactory(mlp);

            var output = forward.ComputeOutput(dataset);

            Assert.IsTrue(output.Count == 1);
            Assert.IsTrue(output[0].State.Length == 1);

            return
                output[0].State[0];
        }

        public float ExecuteTestWith_5_300_1_MLP(
            IDataSet dataset,
            Func<IFunction> functionFactory,
            Func<IMLP, IForwardPropagation> forwardFactory)
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }
            if (functionFactory == null)
            {
                throw new ArgumentNullException("functionFactory");
            }
            if (forwardFactory == null)
            {
                throw new ArgumentNullException("forwardFactory");
            }

            var randomizer = new ConstRandomizer(1f);

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
                    5,
                    300,
                    1
                });

            var forward = forwardFactory(mlp);

            var output = forward.ComputeOutput(dataset);

            Assert.IsTrue(output.Count == 1);
            Assert.IsTrue(output[0].State.Length == 1);

            return
                output[0].State[0];
        }

    }
}