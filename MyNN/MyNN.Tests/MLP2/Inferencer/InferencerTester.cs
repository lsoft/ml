using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.DropConnect.Inferencer;
using MyNN.MLP.DropConnect.Inferencer.CSharp;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.Inferencer
{
    internal class InferencerTester<T>
        where T : ILayerInferencer
    {
        public void Test(
            IDeviceChooser deviceChooser,
            IFunction activationFunction,
            int sampleCount,
            float p,
            out float[] orig,
            out float[] test
            )
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            const int currentLayerNeuronCount = 50;
            const int previousLayerNeuronCount = 50;

            using (var clProvider = new CLProvider(deviceChooser, true))
            {
                var structureRandomizer = new DefaultRandomizer(123);

                var nf = new NeuronFactory(
                    structureRandomizer);

                var layer0 = new Layer(
                    nf,
                    previousLayerNeuronCount
                    );

                var layer1 = new Layer(
                    nf,
                    activationFunction,
                    currentLayerNeuronCount,
                    previousLayerNeuronCount,
                    true,
                    true
                    );

                var previousContainer = new MemLayerContainer(
                    clProvider,
                    layer0.NonBiasNeuronCount,
                    layer1.Neurons.Length
                    );

                previousContainer.StateMem.Array.Fill(j => structureRandomizer.Next());
                previousContainer.StateMem.Write(BlockModeEnum.Blocking);

                var currentContainer = new MemLayerContainer(
                    clProvider,
                    layer0.Neurons.Length,
                    layer1.NonBiasNeuronCount,
                    layer1.Neurons.Length
                    );

                currentContainer.WeightMem.Array.Fill(j => structureRandomizer.Next());
                currentContainer.WeightMem.Write(BlockModeEnum.Blocking);

                const int inferenceRandomizerSeed = 112233;

                //----------------------------------------------------------------------------

                var inf0 = new CorrectInferencer(
                    new DefaultRandomizer(inferenceRandomizerSeed),
                    clProvider,
                    sampleCount,
                    layer0,
                    layer1,
                    currentContainer.WeightMem,
                    previousContainer.StateMem,
                    currentContainer.StateMem,
                    p
                    );

                var before0 = DateTime.Now;

                inf0.InferenceLayer();

                currentContainer.StateMem.Read(BlockModeEnum.Blocking);

                var after0 = DateTime.Now;

                orig = currentContainer.StateMem.Array.CloneArray();

                //----------------------------------------------------------------------------

                var inf1 = (T)Activator.CreateInstance(
                    typeof(T),
                    new DefaultRandomizer(inferenceRandomizerSeed),
                    clProvider,
                    sampleCount,
                    layer0,
                    layer1,
                    currentContainer.WeightMem,
                    previousContainer.StateMem,
                    currentContainer.StateMem,
                    p
                    );

                var before1 = DateTime.Now;

                inf1.InferenceLayer();

                currentContainer.StateMem.Read(BlockModeEnum.Blocking);

                var after1 = DateTime.Now;

                test = currentContainer.StateMem.Array.CloneArray();

                //----------------------------------------------------------------------------

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "1st takes: {0}",
                        (after0 - before0)));

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "2nd takes: {0}",
                        (after1 - before1)));

                if (orig == null)
                {
                    Assert.Fail("orig == null");
                }

                if (test == null)
                {
                    Assert.Fail("test == null");
                }

                if (orig.Length != test.Length)
                {
                    Assert.Fail("orig.Length != test.Length");
                }

            }
        }
    }

    
}
