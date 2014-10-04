﻿using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.LayerPropagator.DropConnect.TrainItemForward.OpenCL.GPU2
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
                MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp;
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
                MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp;
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
                MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp;
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
                MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp;
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
        public void Test0_40_1()
        {
            const uint bitMask = 2;
            Func<int, uint> fillFunc = (i) => ((i % 2) > 0 ? (uint)2 : (uint)0);

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 40;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp;
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
            const uint bitMask = 2;
            Func<int, uint> fillFunc = (i) => ((i % 2) > 0 ? (uint)2 : (uint)0);

            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 40;
            const int currentLayerNeuronCount = 1;

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), false))
            {
                Layer l;
                MemLayerContainer plc;
                MemLayerContainer clc;
                MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp;
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
            uint bitMask,
            Func<int, uint> fillFunc,
            out Layer currentLayer, 
            out MemLayerContainer plc,
            out MemLayerContainer clc,
            out MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator lp)
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

            var ks = new MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.KernelSource();

            var mc = new TestPurposeBitWeightMaskContainer(
                clProvider,
                bitMask,
                previousLayer.GetConfiguration(), 
                currentLayer.GetConfiguration(),
                fillFunc
                );

            lp = new MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.GPU2.LayerPropagator(
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
