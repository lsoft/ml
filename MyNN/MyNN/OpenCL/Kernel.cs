using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.OpenCL
{
    public class Kernel : IDisposable
    {
        private readonly Cl.CommandQueue _commandQueue;
        private readonly Cl.Program _program;
        private Cl.Kernel _kernel;

        public Kernel(
            Cl.CommandQueue commandQueue,
            Cl.Program program,
            string kernelName)
        {
            _commandQueue = commandQueue;
            _program = program;
            
            _kernel = _program.CreateKernel(kernelName);
        }

        public Kernel SetKernelArgMem<T>(uint argId, Mem<T> mem)
            where T : struct
        {
            _kernel.SetKernelArgMem(argId, mem.GetMem());

            return this;
        }

        public Kernel SetKernelArg(uint argId, int size, object data)
        {
            _kernel.SetKernelArg(argId, size, data);

            return this;
        }

        public void EnqueueNDRangeKernel(params int[] sizes)
        {
            _kernel.EnqueueNDRangeKernel(_commandQueue, sizes);
        }


        //public void Execute()
        //{
        //    _kernel.EnqueueTask(_commandQueue);
        //}

        public void Dispose()
        {
            Cl.ReleaseKernel(_kernel);
            Cl.ReleaseProgram(_program);
        }
    }
}
