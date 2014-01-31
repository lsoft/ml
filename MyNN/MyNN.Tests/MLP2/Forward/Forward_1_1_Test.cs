using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward
{
    internal class Forward_1_1_Test
    {

        public float ExecuteTest(
            DataSet dataset,
            float weight0,
            float weight1,
            Func<IFunction> functionFactory,
            Func<CLProvider, MLP, IForwardPropagation> forwardFactory)
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
                var forward = forwardFactory(clProvider, mlp);

                var output = forward.ComputeOutput(dataset);

                Assert.IsTrue(output.Count == 1);
                Assert.IsTrue(output[0].State.Length == 1);

                return
                    output[0].State[0];
            }

        }
    }
}