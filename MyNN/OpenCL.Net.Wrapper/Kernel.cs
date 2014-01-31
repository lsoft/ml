using System;
using System.Linq;
using OpenCL.Net.Wrapper.Mem;

namespace OpenCL.Net.Wrapper
{
    public class Kernel : IDisposable
    {
        private readonly CommandQueue _commandQueue;
        private readonly Program _program;
        private readonly OpenCL.Net.Kernel _kernel;

        public Kernel(
            CommandQueue commandQueue,
            Program program,
            string kernelName)
        {
            _commandQueue = commandQueue;
            _program = program;


            ErrorCode errorCode;
            _kernel = Cl.CreateKernel(_program, kernelName, out errorCode);

            if (errorCode != ErrorCode.Success)
            {
                throw new Cl.Exception(errorCode);
            }

            //_kernel = _program.CreateKernel(kernelName);
        }

        public Kernel SetKernelArgLocalMem(uint argId, int size)
        {
            var error = Cl.SetKernelArg(_kernel, argId, new IntPtr(size), null);

            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.SetKernelArgMem: {0}!", error));
            }

            //_kernel.SetKernelArgLocalMem(argId, size);

            return this;
        }

        public Kernel SetKernelArgMem<T>(uint argId, OpenCL.Net.Wrapper.Mem.Mem<T> mem)
            where T : struct
        {
            if (mem == null)
            {
                throw new ArgumentNullException("mem");
            }

            var error = Cl.SetKernelArg(_kernel, argId, mem.GetMem());

            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.SetKernelArgMem: {0}!", error));
            }


            //_kernel.SetKernelArgMem(argId, mem.GetMem());

            return this;
        }

        public Kernel SetKernelArg(uint argId, int size, object data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var error = Cl.SetKernelArg(_kernel, argId, new IntPtr(size), data);
            
            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.SetKernelArg: {0}!", error));
            }

            //_kernel.SetKernelArg(argId, size, data);

            return this;
        }

        /// <summary>
        /// without local work sizes
        /// </summary>
        public void EnqueueNDRangeKernel(params int[] sizes)
        {
            Event clevent;

            var error = Cl.EnqueueNDRangeKernel(
                _commandQueue,
                _kernel,
                (uint)sizes.Length,
                null,
                sizes.ToList().Select(size => new IntPtr(size)).ToArray(),
                null,
                0,
                null,
                out clevent);

            if (error != ErrorCode.Success)
            {
                throw new InvalidOperationException("EnqueueNDRangeKernel failed:" + error);
            }

            clevent.Dispose();

            //_kernel.EnqueueNDRangeKernel(_commandQueue, sizes);
        }

        /// <summary>
        /// with local work sizes
        /// </summary>
        public void EnqueueNDRangeKernel(int[] globalSizes, int[] localSizes)
        {
            Event clevent;

            if (globalSizes.Length != localSizes.Length)
            {
                throw new InvalidOperationException("globalSizes.Length != localSizes.Length");
            }

            ErrorCode error = Cl.EnqueueNDRangeKernel(
                _commandQueue,
                _kernel,
                (uint)globalSizes.Length,
                null,
                globalSizes.ToList().Select(size => new IntPtr(size)).ToArray(),
                localSizes.ToList().Select(size => new IntPtr(size)).ToArray(),
                0,
                null,
                out clevent);

            if (error != ErrorCode.Success)
            {
                throw new InvalidOperationException("EnqueueNDRangeKernel failed:" + error);
            }

            clevent.Dispose();

            //_kernel.EnqueueNDRangeKernel(_commandQueue, globalSizes, localSizes);
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
