using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace OpenCL.Net
{
    /// <summary>
    /// Extension class methods for Cloo
    /// </summary>
    public static class OpenCLExtensions
    {
        private static int _intPtrSize = -1;

        public static int IntPtrSize
        {
            get
            {
                if (_intPtrSize == -1)
                {
                    unsafe
                    {
                        _intPtrSize = sizeof(IntPtr);
                    }
                }
                return _intPtrSize;
            }
        }

        #region kernel

        public static Cl.Kernel CreateKernel(this Cl.Program program, string kernelName)
        {
            Cl.ErrorCode error;
            Cl.Kernel kernel = Cl.CreateKernel(program, kernelName, out error);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.CreateKernel for {1}: {0}!", error, kernelName));
            }
            return kernel;
        }

        public static Cl.Kernel SetKernelArg(this Cl.Kernel kernel, uint argId, int size, object data)
        {
            Cl.ErrorCode error = Cl.SetKernelArg(kernel, argId, size, data);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.SetKernelArg: {0}!", error));
            }
            return kernel;
        }

        public static Cl.Kernel SetKernelArgMem(this Cl.Kernel kernel, uint argId, Cl.Mem mem)
        {
            Cl.ErrorCode error = Cl.SetKernelArg(kernel, argId, IntPtrSize, mem);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.SetKernelArgMem: {0}!", error));
            }
            return kernel;
        }

        public static void EnqueueNDRangeKernel(this Cl.Kernel kernel, Cl.CommandQueue commandQueue, params int[] sizes)
        {
            Cl.Event clevent;
            Cl.ErrorCode error = Cl.EnqueueNDRangeKernel(
                commandQueue,
                kernel,
                (uint)sizes.Length,
                null,
                sizes.ToList().Select(size => new IntPtr(size)).ToArray(),
                null,
                0,
                null,
                out clevent);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidOperationException("EnqueueNDRangeKernel failed:" + error);
            }
            clevent.Dispose();
        }

        public static void EnqueueTask(this Cl.Kernel kernel, Cl.CommandQueue commandQueue)
        {
            Cl.Event clevent;
            Cl.ErrorCode error = Cl.EnqueueTask(
                commandQueue,
                kernel,
                0,
                null,
                out clevent);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidOperationException("EnqueueTask failed:" + error);
            }
            clevent.Dispose();
        }

        #endregion

        #region mem

        public static void Read(this Cl.Mem mem, Cl.CommandQueue commandQueue, int size, object data, BlockModeEnum blockMode)
        {
            Cl.Event clevent;
            Cl.ErrorCode error = Cl.EnqueueReadBuffer(commandQueue, mem, blockMode == BlockModeEnum.Blocking ? Cl.Bool.True : Cl.Bool.False, IntPtr.Zero, new IntPtr(size), data, 0, null, out clevent);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidOperationException("EnqueueReadBuffer failed for _weightsMem:" + error);
            }
            clevent.Dispose();
        }

        public static void ReadDoubleArray(this Cl.Mem mem, Cl.CommandQueue commandQueue, double[] array, BlockModeEnum blockMode)
        {
            mem.Read(commandQueue, sizeof(double) * array.Length, array, blockMode);
        }

        public static void ReadFloatArray(this Cl.Mem mem, Cl.CommandQueue commandQueue, float[] array, BlockModeEnum blockMode)
        {
            mem.Read(commandQueue, sizeof(float) * array.Length, array, blockMode);
        }

        public static void Write(this Cl.Mem mem, Cl.CommandQueue commandQueue, int size, object data, BlockModeEnum blockMode)
        {
            Cl.Event clevent;
            Cl.ErrorCode error = Cl.EnqueueWriteBuffer(commandQueue, mem, blockMode == BlockModeEnum.Blocking ? Cl.Bool.True : Cl.Bool.False, IntPtr.Zero, new IntPtr(size), data, 0, null, out clevent);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("EnqueueWriteBuffer failed: {0}!", error));
            }
            clevent.Dispose();
        }

        public static void WriteDoubleArray(this Cl.Mem mem, Cl.CommandQueue commandQueue, double[] array, BlockModeEnum blockMode)
        {
            mem.Write(commandQueue, sizeof(double) * array.Length, array, blockMode);
        }

        public static void WriteFloatArray(this Cl.Mem mem, Cl.CommandQueue commandQueue, float[] array, BlockModeEnum blockMode)
        {
            mem.Write(commandQueue, sizeof(float) * array.Length, array, blockMode);
        }

        public static Cl.Mem AllocateArray<T>(
            this Cl.Context context, 
            T[] array, 
            int sizeofT,
            Cl.MemFlags flags)
            where T : struct
        {
            var size = sizeofT * array.Count();

            Cl.ErrorCode error;
            var result = Cl.CreateBuffer(context, flags, (IntPtr)size, array, out error);

            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("CreateBuffer failed:", error));
            }

            return result;
        }

        //public static Cl.Mem AllocateDoubleArray(this Cl.Context context, double[] array, Cl.MemFlags flags)
        //{
        //    Cl.ErrorCode error;
        //    int size = sizeof(double) * array.Count();
        //    Cl.Mem result = Cl.CreateBuffer(context, flags, (IntPtr)size, array, out error);
        //    if (error != Cl.ErrorCode.Success)
        //    {
        //        throw new InvalidProgramException(string.Format("CreateBuffer failed:", error));
        //    }
        //    return result;
        //}

        //public static Cl.Mem AllocateFloatArray(this Cl.Context context, float[] array, Cl.MemFlags flags)
        //{
        //    Cl.ErrorCode error;
        //    int size = sizeof(float) * array.Count();
        //    Cl.Mem result = Cl.CreateBuffer(context, flags, (IntPtr)size, array, out error);
        //    if (error != Cl.ErrorCode.Success)
        //    {
        //        throw new InvalidProgramException(string.Format("CreateBuffer failed:", error));
        //    }
        //    return result;
        //}

        //public static Cl.Mem AllocateIntArray(this Cl.Context context, int[] array, Cl.MemFlags flags)
        //{
        //    Cl.ErrorCode error;
        //    int size = 4 * array.Count();
        //    Cl.Mem result = Cl.CreateBuffer(context, flags, (IntPtr)size, array, out error);
        //    if (error != Cl.ErrorCode.Success)
        //    {
        //        throw new InvalidProgramException(string.Format("CreateBuffer failed:", error));
        //    }
        //    return result;
        //}

        #endregion

        #region command queue

        public static void Finish(this Cl.CommandQueue commandQueue)
        {
            Cl.ErrorCode error = Cl.Finish(commandQueue);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Finish failed:", error));
            }
        }

        #endregion

    }
}