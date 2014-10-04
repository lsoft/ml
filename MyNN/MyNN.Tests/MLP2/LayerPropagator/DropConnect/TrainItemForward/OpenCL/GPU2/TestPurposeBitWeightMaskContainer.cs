using System;
using MyNN.MLP2.ForwardPropagation.DropConnect.WeightMaskContainer2;
using MyNN.MLP2.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.Tests.MLP2.LayerPropagator.DropConnect.TrainItemForward.OpenCL.GPU2
{
    internal class TestPurposeBitWeightMaskContainer : IOpenCLWeightBitMaskContainer2
    {
        public uint BitMask
        {
            get;
            private set;
        }

        public MemUint MaskMem
        {
            get;
            private set;
        }

        public TestPurposeBitWeightMaskContainer(
            CLProvider clProvider,
            uint bitMask,
            ILayerConfiguration previousLayerConfiguration,
            ILayerConfiguration currentLayerConfiguration,
            Func<int, uint> fillFunc
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }
            if (fillFunc == null)
            {
                throw new ArgumentNullException("fillFunc");
            }

            BitMask = bitMask;

            MaskMem = clProvider.CreateUintMem(
                currentLayerConfiguration.NonBiasNeuronCount * previousLayerConfiguration.Neurons.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            MaskMem.Array.Fill(fillFunc);

            MaskMem.Write(BlockModeEnum.Blocking);
        }

        public void RegenerateMask()
        {
            //nothing to do
        }
    }
}