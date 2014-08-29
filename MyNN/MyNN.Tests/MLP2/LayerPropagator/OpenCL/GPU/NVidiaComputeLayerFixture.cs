using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.ForwardPropagation.Classic.CSharp;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.LayerPropagator.OpenCL.GPU
{
    [TestClass]
    public class NVidiaComputeLayerFixture
    {
        private const float Epsilon = 1e-6f;

        [TestMethod]
        public void Test_1_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 1;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                GPULayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    out l,
                    out plc,
                    out clc,
                    out lp);

                l.Neurons[0].Weights[0] = 0.5f;
                plc.NetMem.Array[0] = 1f;
                plc.StateMem.Array[0] = 1f;

                plc.PushHiddenLayers();

                clc.ClearAndPushHiddenLayers();
                clc.PushWeights(l);
                clProvider.QueueFinish();

                lp.ComputeLayer();
                lp.WaitForCalculationFinished();

                clc.StateMem.Read(BlockModeEnum.Blocking);

                const float CorrectValue = 0.5f;

                Assert.IsTrue(clc.StateMem.Array[0].IsEquals(CorrectValue, Epsilon));
            }
        }

        [TestMethod]
        public void Test_2_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 2;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                GPULayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    out l,
                    out plc,
                    out clc,
                    out lp);

                l.Neurons[0].Weights[0] = 0.5f;
                l.Neurons[0].Weights[1] = -0.5f;

                plc.NetMem.Array[0] = -2f;
                plc.StateMem.Array[0] = -2f;
                plc.NetMem.Array[1] = 1f;
                plc.StateMem.Array[1] = 1f;

                plc.PushHiddenLayers();

                clc.ClearAndPushHiddenLayers();
                clc.PushWeights(l);
                clProvider.QueueFinish();

                lp.ComputeLayer();
                lp.WaitForCalculationFinished();

                clc.StateMem.Read(BlockModeEnum.Blocking);

                const float CorrectValue = -1.5f;

                Assert.IsTrue(clc.StateMem.Array[0].IsEquals(CorrectValue, Epsilon));
            }
        }

        [TestMethod]
        public void Test0_40_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 40;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                GPULayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    out l,
                    out plc,
                    out clc,
                    out lp);

                l.Neurons[0].Weights.Fill((a) => (float) a);

                plc.NetMem.Array.Fill(1f);
                plc.StateMem.Array.Fill(1f);

                plc.PushHiddenLayers();

                clc.ClearAndPushHiddenLayers();
                clc.PushWeights(l);
                clProvider.QueueFinish();

                lp.ComputeLayer();
                lp.WaitForCalculationFinished();

                clc.StateMem.Read(BlockModeEnum.Blocking);

                float correctValue = Enumerable.Range(0, previousLayerNeuronCount).Sum();

                Assert.IsTrue(clc.StateMem.Array[0].IsEquals(correctValue, Epsilon));
            }
        }

        [TestMethod]
        public void Test1_40_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 40;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                GPULayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    out l,
                    out plc,
                    out clc,
                    out lp);

                l.Neurons[0].Weights.Fill((a) => (float) a);

                plc.NetMem.Array.Fill((a) => (float) a);
                plc.StateMem.Array.Fill((a) => (float) a);

                plc.PushHiddenLayers();

                clc.ClearAndPushHiddenLayers();
                clc.PushWeights(l);
                clProvider.QueueFinish();

                lp.ComputeLayer();
                lp.WaitForCalculationFinished();

                clc.StateMem.Read(BlockModeEnum.Blocking);

                var correctArray = Enumerable.Range(0, previousLayerNeuronCount).ToArray();
                correctArray.Transform((a) => a*a);
                float correctValue = correctArray.Sum();

                Assert.IsTrue(clc.StateMem.Array[0].IsEquals(correctValue, Epsilon));
            }
        }

        private static void ConstuctComponents(
            CLProvider clProvider,
            INeuronFactory nf, 
            int previousLayerNeuronCount, 
            int currentLayerNeuronCount,
            out Layer l, 
            out MemLayerContainer plc,
            out MemLayerContainer clc, 
            out GPULayerPropagator lp)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (nf == null)
            {
                throw new ArgumentNullException("nf");
            }

            var function = new LinearFunction(1f);

            l = new Layer(
                nf,
                function,
                currentLayerNeuronCount,
                previousLayerNeuronCount,
                false,
                false
                );

            plc = new MemLayerContainer(
                clProvider,
                previousLayerNeuronCount,
                previousLayerNeuronCount
                );

            clc = new MemLayerContainer(
                clProvider,
                previousLayerNeuronCount,
                currentLayerNeuronCount,
                currentLayerNeuronCount
                );

            var ks = new GPUKernelSource();

            lp = new GPULayerPropagator(
                clProvider,
                ks,
                plc,
                clc,
                function,
                previousLayerNeuronCount,
                currentLayerNeuronCount
                );
        }
    }
}
