using System;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL
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

        public Kernel SetKernelArgLocalMem(uint argId, int size)
        {
            _kernel.SetKernelArgLocalMem(argId, size);

            return this;
        }

        public Kernel SetKernelArgMem<T>(uint argId, Mem<T> mem)
            where T : struct
        {
            if (mem == null)
            {
                throw new ArgumentNullException("mem");
            }

            _kernel.SetKernelArgMem(argId, mem.GetMem());

            return this;
        }

        public Kernel SetKernelArg(uint argId, int size, object data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            _kernel.SetKernelArg(argId, size, data);

            return this;
        }

        /// <summary>
        /// without local work sizes
        /// </summary>
        public void EnqueueNDRangeKernel(params int[] sizes)
        {
            _kernel.EnqueueNDRangeKernel(_commandQueue, sizes);
        }

        /// <summary>
        /// with local work sizes
        /// </summary>
        public void EnqueueNDRangeKernel(int[] globalSizes, int[] localSizes)
        {
            _kernel.EnqueueNDRangeKernel(_commandQueue, globalSizes, localSizes);
        }

        public void Execute()
        {
            _kernel.EnqueueTask(_commandQueue);
        }

        public void Dispose()
        {
            Cl.ReleaseKernel(_kernel);
            Cl.ReleaseProgram(_program);
        }
    }
}
