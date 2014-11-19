using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.LayerPropagator.DropConnect.OpenCL.GPU2
{
    [TestClass]
    public class IntelCPUComputeLayerFixture
    {
        private const float Epsilon = 1e-6f;

        [TestMethod]
        public void Test_1_1_WithMask0()
        {
            const uint bitMask = 2;
            Func<int, uint> fillFunc = (i) => (uint)1;

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 1;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                DropConnectLayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    bitMask,
                    fillFunc,
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

                const float CorrectResult = 0.0f;
                float result = clc.StateMem.Array[0]; 

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Result = {0}, correct result = {1}",
                        result,
                        CorrectResult
                        ));

                Assert.IsTrue(result.IsEquals(CorrectResult, Epsilon));
            }
        }

        [TestMethod]
        public void Test_1_1_WithMask1()
        {
            const uint bitMask = 1;
            Func<int, uint> fillFunc = (i) => (uint)0;

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 1;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                DropConnectLayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    bitMask,
                    fillFunc,
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

                const float CorrectResult = 0.0f;
                float result = clc.StateMem.Array[0];

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Result = {0}, correct result = {1}",
                        result,
                        CorrectResult
                        ));

                Assert.IsTrue(result.IsEquals(CorrectResult, Epsilon));
            }
        }

        [TestMethod]
        public void Test_1_1_WithoutMask()
        {
            const uint bitMask = 2;
            Func<int, uint> fillFunc = (i) => (uint)2;

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 1;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                DropConnectLayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    bitMask,
                    fillFunc,
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

                const float CorrectResult = 0.5f;

                float result = clc.StateMem.Array[0];

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Result = {0}, correct result = {1}",
                        result,
                        CorrectResult
                        ));

                Assert.IsTrue(result.IsEquals(CorrectResult, Epsilon));
            }
        }

        [TestMethod]
        public void Test_2_1_WithoutMask()
        {
            const uint bitMask = 2;
            Func<int, uint> fillFunc = (i) => (uint)2;

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 2;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                DropConnectLayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    bitMask,
                    fillFunc,
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

                const float CorrectResult = -1.5f;

                float result = clc.StateMem.Array[0];

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Result = {0}, correct result = {1}",
                        result,
                        CorrectResult
                        ));

                Assert.IsTrue(result.IsEquals(CorrectResult, Epsilon));
            }
        }

        [TestMethod]
        public void Test_2_1_WithMask()
        {
            var mask = new[] { (uint)1, (uint)2, (uint)1 };

            const uint bitMask = 2;
            Func<int, uint> fillFunc = (i) => mask[i];

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 2;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                DropConnectLayerPropagator lp;
                ConstuctComponents(
                    clProvider,
                    nf,
                    previousLayerNeuronCount,
                    currentLayerNeuronCount,
                    bitMask,
                    fillFunc,
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

                const float CorrectResult = -0.5f;

                float result = clc.StateMem.Array[0];

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Result = {0}, correct result = {1}",
                        result,
                        CorrectResult
                        ));

                Assert.IsTrue(result.IsEquals(CorrectResult, Epsilon));
            }
        }

        private static void ConstuctComponents(
            CLProvider clProvider,
            INeuronFactory nf, 
            int previousLayerNeuronCount, 
            int currentLayerNeuronCount,
            uint bitMask,
            Func<int, uint> fillFunc,
            out Layer currentLayer, 
            out MemLayerContainer plc,
            out MemLayerContainer clc,
            out DropConnectLayerPropagator lp)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (nf == null)
            {
                throw new ArgumentNullException("nf");
            }
            if (fillFunc == null)
            {
                throw new ArgumentNullException("fillFunc");
            }

            var function = new LinearFunction(1f);

            var previousLayer = new Layer(
                nf,
                previousLayerNeuronCount
                );

            currentLayer = new Layer(
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

            var ks = new KernelSource();

            var mc = new TestPurposeMaskContainer(
                clProvider,
                bitMask,
                previousLayer.GetConfiguration(), 
                currentLayer.GetConfiguration(),
                fillFunc
                );

            lp = new DropConnectLayerPropagator(
                clProvider,
                ks,
                mc,
                plc,
                clc,
                function,
                previousLayerNeuronCount,
                currentLayerNeuronCount
                );
        }
    }
}
