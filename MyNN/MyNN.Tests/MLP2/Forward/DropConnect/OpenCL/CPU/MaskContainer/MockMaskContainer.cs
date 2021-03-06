﻿using System;
using MyNN.Common.Other;
using MyNN.Mask;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.Tests.MLP2.Forward.DropConnect.OpenCL.CPU.MaskContainer
{
    internal class MockMaskContainer : IOpenCLMaskContainer
    {
        public MockMaskContainer(
            CLProvider clProvider,
            ILayerConfiguration previousLayerConfiguration,
            ILayerConfiguration currentLayerConfiguration,
            uint bitMask,
            Func<int, uint> layerMasks
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
            if (layerMasks == null)
            {
                throw new ArgumentNullException("layerMasks");
            }

            this.BitMask = bitMask;

            this.MaskMem = clProvider.CreateUintMem(
                currentLayerConfiguration.TotalNeuronCount * previousLayerConfiguration.TotalNeuronCount, //without bias neuron at current layer, but include bias neuron at previous layer
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            this.MaskMem.Array.Fill(layerMasks);

            this.MaskMem.Write(BlockModeEnum.Blocking);
        }

        public MockMaskContainer(
            CLProvider clProvider,
            ILayerConfiguration previousLayerConfiguration,
            ILayerConfiguration currentLayerConfiguration,
            uint bitMask,
            uint mask
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

            this.BitMask = bitMask;

            this.MaskMem = clProvider.CreateUintMem(
                currentLayerConfiguration.TotalNeuronCount * previousLayerConfiguration.TotalNeuronCount, //without bias neuron at current layer, but include bias neuron at previous layer
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            this.MaskMem.Array.Fill(mask);

            this.MaskMem.Write(BlockModeEnum.Blocking);
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
        
        public MemUint MaskMem
        {
            get;
            private set;
        }
    }
}
