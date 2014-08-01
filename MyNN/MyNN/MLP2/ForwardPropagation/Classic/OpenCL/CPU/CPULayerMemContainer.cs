using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU
{
    public class CPULayerMemContainer
    {
        public MemFloat WeightMem
        {
            get;
            private set;
        }

        public MemFloat NetMem
        {
            get;
            private set;
        }

        public MemFloat StateMem
        {
            get;
            private set;
        }

        public CPULayerMemContainer(
            CLProvider clProvider,
            int previousLayerNeuronCount,
            int currentLayerNeuronCount
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            //нейроны
            var netMem = clProvider.CreateFloatMem(
                currentLayerNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            netMem.Write(BlockModeEnum.Blocking);

            NetMem = netMem;

            var stateMem = clProvider.CreateFloatMem(
                currentLayerNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            stateMem.Write(BlockModeEnum.Blocking);

            StateMem = stateMem;

            //веса

            var weightMem = clProvider.CreateFloatMem(
                currentLayerNeuronCount*previousLayerNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            weightMem.Write(BlockModeEnum.Blocking);
            
            WeightMem = weightMem;
        }


    }
}
