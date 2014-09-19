using System;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.WeightMask;
using MyNN.MLP2.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.Tests.MLP2.Forward.DropConnect.TrainItemForward.CPU.Float.MaskContainer
{
    internal class MockWeightMaskContainer : IOpenCLWeightMaskContainer
    {
        public MockWeightMaskContainer(
            CLProvider clProvider,
            IMLP mlp,
            Func<int, int, float> layerMasks
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (layerMasks == null)
            {
                throw new ArgumentNullException("layerMasks");
            }


            var layerCount = mlp.Layers.Length;

            this.MaskMem = new MemFloat[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                this.MaskMem[layerIndex] = clProvider.CreateFloatMem(
                    mlp.Layers[layerIndex].NonBiasNeuronCount * mlp.Layers[layerIndex].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadOnly);

                this.MaskMem[layerIndex].Array.Fill(
                    (weightIndex) => layerMasks(layerIndex, weightIndex));

                this.MaskMem[layerIndex].Write(BlockModeEnum.Blocking);
            }
        }

        public MockWeightMaskContainer(
            CLProvider clProvider,
            IMLP mlp,
            float[] layerMasks
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (layerMasks == null)
            {
                throw new ArgumentNullException("layerMasks");
            }

            var layerCount = mlp.Layers.Length;

            this.MaskMem = new MemFloat[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                this.MaskMem[layerIndex] = clProvider.CreateFloatMem(
                    mlp.Layers[layerIndex].NonBiasNeuronCount * mlp.Layers[layerIndex].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadOnly);

                this.MaskMem[layerIndex].Array.Fill(layerMasks[layerIndex]);

                this.MaskMem[layerIndex].Write(BlockModeEnum.Blocking);
            }
        }

        public MockWeightMaskContainer(
            CLProvider clProvider,
            IMLP mlp,
            float mask
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var layerCount = mlp.Layers.Length;

            this.MaskMem = new MemFloat[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                this.MaskMem[layerIndex] = clProvider.CreateFloatMem(
                    mlp.Layers[layerIndex].NonBiasNeuronCount * mlp.Layers[layerIndex].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadOnly);

                this.MaskMem[layerIndex].Array.Fill(mask);

                this.MaskMem[layerIndex].Write(BlockModeEnum.Blocking);
            }
        }

        public void RegenerateMask()
        {
            //nothing to do
        }
        
        public MemFloat[] MaskMem
        {
            get;
            private set;
        }
    }
}
