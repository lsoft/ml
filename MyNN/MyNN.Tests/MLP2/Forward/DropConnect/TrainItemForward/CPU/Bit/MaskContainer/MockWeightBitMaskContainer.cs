using System;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask;
using MyNN.MLP2.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.Forward.DropConnect.TrainItemForward.CPU.Bit.MaskContainer
{
    internal class MockWeightBitMaskContainer : IOpenCLWeightBitMaskContainer
    {
        public MockWeightBitMaskContainer(
            CLProvider clProvider,
            IMLP mlp,
            uint bitMask,
            Func<int, int, uint> layerMasks
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

            this.BitMask = bitMask;

            var layerCount = mlp.Layers.Length;

            this.MaskMem = new MemUint[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                this.MaskMem[layerIndex] = clProvider.CreateUintMem(
                    mlp.Layers[layerIndex].NonBiasNeuronCount * mlp.Layers[layerIndex].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadOnly);

                this.MaskMem[layerIndex].Array.Fill(
                    (weightIndex) => layerMasks(layerIndex, weightIndex));

                this.MaskMem[layerIndex].Write(BlockModeEnum.Blocking);
            }
        }

        public MockWeightBitMaskContainer(
            CLProvider clProvider,
            IMLP mlp,
            uint bitMask,
            uint[] layerMasks
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

            this.BitMask = bitMask;

            var layerCount = mlp.Layers.Length;

            this.MaskMem = new MemUint[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                this.MaskMem[layerIndex] = clProvider.CreateUintMem(
                    mlp.Layers[layerIndex].NonBiasNeuronCount * mlp.Layers[layerIndex].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadOnly);

                this.MaskMem[layerIndex].Array.Fill(layerMasks[layerIndex]);

                this.MaskMem[layerIndex].Write(BlockModeEnum.Blocking);
            }
        }

        public MockWeightBitMaskContainer(
            CLProvider clProvider,
            IMLP mlp,
            uint bitMask,
            uint mask
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

            this.BitMask = bitMask;

            var layerCount = mlp.Layers.Length;

            this.MaskMem = new MemUint[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                this.MaskMem[layerIndex] = clProvider.CreateUintMem(
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

        public uint BitMask
        {
            get;
            private set;
        }
        
        public MemUint[] MaskMem
        {
            get;
            private set;
        }
    }
}
