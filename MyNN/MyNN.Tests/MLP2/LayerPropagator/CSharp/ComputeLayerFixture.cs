﻿using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.ForwardPropagation.Classic.CSharp;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.Tests.MLP2.LayerPropagator.CSharp
{
    [TestClass]
    public class ComputeLayerFixture
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

            Layer l;
            CSharpLayerContainer plc;
            CSharpLayerContainer clc;
            CSharpLayerPropagator lp;
            ConstuctComponents(
                nf, 
                previousLayerNeuronCount, 
                currentLayerNeuronCount, 
                out l, 
                out plc,
                out clc,
                out lp);

            l.Neurons[0].Weights[0] = 0.5f;
            plc.NetMem[0] = 1f;
            plc.StateMem[0] = 1f;

            lp.ComputeLayer();
            lp.WaitForCalculationFinished();

            const float CorrectValue = 0.5f;

            Assert.IsTrue(clc.StateMem[0].IsEquals(CorrectValue, Epsilon));
            
        }

        [TestMethod]
        public void Test_2_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 2;
            const int currentLayerNeuronCount = 1;

            Layer l;
            CSharpLayerContainer plc;
            CSharpLayerContainer clc;
            CSharpLayerPropagator lp;
            ConstuctComponents(
                nf,
                previousLayerNeuronCount,
                currentLayerNeuronCount,
                out l,
                out plc,
                out clc,
                out lp);

            l.Neurons[0].Weights[0] = 0.5f;
            l.Neurons[0].Weights[1] = -0.5f;
            
            plc.NetMem[0] = -2f;
            plc.StateMem[0] = -2f;
            plc.NetMem[1] = 1f;
            plc.StateMem[1] = 1f;

            lp.ComputeLayer();
            lp.WaitForCalculationFinished();

            const float CorrectValue = -1.5f;

            Assert.IsTrue(clc.StateMem[0].IsEquals(CorrectValue, Epsilon));

        }

        [TestMethod]
        public void Test0_40_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 40;
            const int currentLayerNeuronCount = 1;

            Layer l;
            CSharpLayerContainer plc;
            CSharpLayerContainer clc;
            CSharpLayerPropagator lp;
            ConstuctComponents(
                nf,
                previousLayerNeuronCount,
                currentLayerNeuronCount,
                out l,
                out plc,
                out clc,
                out lp);

            l.Neurons[0].Weights.Fill((a) => (float)a);

            plc.NetMem.Fill(1f);
            plc.StateMem.Fill(1f);

            lp.ComputeLayer();
            lp.WaitForCalculationFinished();

            float correctValue = Enumerable.Range(0, previousLayerNeuronCount).Sum();

            Assert.IsTrue(clc.StateMem[0].IsEquals(correctValue, Epsilon));

        }

        [TestMethod]
        public void Test1_40_1()
        {
            var randomizer = new ConstRandomizer(0.5f);

            var nf = new NeuronFactory(
                randomizer);

            const int previousLayerNeuronCount = 40;
            const int currentLayerNeuronCount = 1;

            Layer l;
            CSharpLayerContainer plc;
            CSharpLayerContainer clc;
            CSharpLayerPropagator lp;
            ConstuctComponents(
                nf,
                previousLayerNeuronCount,
                currentLayerNeuronCount,
                out l,
                out plc,
                out clc,
                out lp);

            l.Neurons[0].Weights.Fill((a) => (float)a);

            plc.NetMem.Fill((a) => (float)a);
            plc.StateMem.Fill((a) => (float)a);

            lp.ComputeLayer();
            lp.WaitForCalculationFinished();

            var correctArray = Enumerable.Range(0, previousLayerNeuronCount).ToArray();
            correctArray.Transform((a) => a * a);
            float correctValue = correctArray.Sum();

            Assert.IsTrue(clc.StateMem[0].IsEquals(correctValue, Epsilon));

        }

        private static void ConstuctComponents(
            INeuronFactory nf, 
            int previousLayerNeuronCount, 
            int currentLayerNeuronCount, 
            out Layer l, 
            out CSharpLayerContainer plc,
            out CSharpLayerContainer clc, 
            out CSharpLayerPropagator lp)
        {
            if (nf == null)
            {
                throw new ArgumentNullException("nf");
            }

            l = new Layer(
                nf,
                new LinearFunction(1f),
                currentLayerNeuronCount,
                previousLayerNeuronCount,
                false,
                false
                );

            plc = new CSharpLayerContainer(
                previousLayerNeuronCount,
                previousLayerNeuronCount
                );

            clc = new CSharpLayerContainer(
                previousLayerNeuronCount,
                currentLayerNeuronCount,
                currentLayerNeuronCount
                );

            lp = new CSharpLayerPropagator(
                l,
                plc,
                clc
                );
        }
    }
}