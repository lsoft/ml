using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.Container;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;


using InferenceAlias = MyNN.MLP2.ForwardPropagation.DropConnect.Inference;

namespace MyNN.Tests.MLP2.Forward.DropConnect.Inferencer
{
    internal class InferencerTester<T>
        where T : InferenceAlias.ILayerInference
    {
        public void Test(
            IFunction activationFunction,
            int sampleCount,
            float p,
            out float[] orig,
            out float[] test
            )
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            const int currentLayerNeuronCount = 50;
            const int previousLayerNeuronCount = 50;

            using (var clProvider = new CLProvider())
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

                var inf0 = new InferenceAlias.CSharp.CorrectInferencer(
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

                inf0.InferenceLayer();
                
                currentContainer.StateMem.Read(BlockModeEnum.Blocking);

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

                inf1.InferenceLayer();

                currentContainer.StateMem.Read(BlockModeEnum.Blocking);

                test = currentContainer.StateMem.Array.CloneArray();

                //----------------------------------------------------------------------------

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
