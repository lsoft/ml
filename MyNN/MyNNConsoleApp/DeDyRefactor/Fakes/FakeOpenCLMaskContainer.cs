using System;
using MyNN.Mask;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNNConsoleApp.DeDyRefactor.Fakes
{
    internal class FakeOpenCLMaskContainer : IOpenCLMaskContainer
    {
        private readonly object _locker = new object();

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

        public FakeOpenCLMaskContainer(
            CLProvider clProvider,
            long arraySize
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            this.BitMask = 1;

            MaskMem = clProvider.CreateUintMem(
                arraySize,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            for (var cc = 0; cc < this.MaskMem.Array.Length; cc++)
            {
                this.MaskMem.Array[cc] = (uint)cc;
            }

            this.RegenerateMask();
        }

        public void RegenerateMask()
        {
            lock (_locker)
            {
                var f = this.MaskMem.Array[0];

                for (var cc = 0; cc < this.MaskMem.Array.Length - 1; cc++)
                {
                    this.MaskMem.Array[cc] = this.MaskMem.Array[cc + 1];
                }

                this.MaskMem.Array[this.MaskMem.Array.Length - 1] = f;

                this.MaskMem.Write(BlockModeEnum.Blocking);
            }
        }

    }
}