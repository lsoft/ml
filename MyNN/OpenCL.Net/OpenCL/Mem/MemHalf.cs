using System;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.Mem
{
    public class MemHalf : Mem<Half>
    {
        public MemHalf(
            Cl.CommandQueue commandQueue,
            Cl.Context context,
            int arrayLength,
            Cl.MemFlags flags)
            : base(commandQueue, context, arrayLength, 2, flags)
        {
        }
    }
}