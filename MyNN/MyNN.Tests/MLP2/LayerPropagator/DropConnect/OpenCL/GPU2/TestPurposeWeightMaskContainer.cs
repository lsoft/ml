using System;
using MyNN.Common.Other;
using MyNN.MLP.DropConnect.WeightMask;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.Tests.MLP2.LayerPropagator.DropConnect.OpenCL.GPU2
{
    internal class TestPurposeWeightMaskContainer : IOpenCLWeightMaskContainer
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

        public TestPurposeWeightMaskContainer(
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