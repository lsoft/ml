using System;


namespace OpenCL.Net.Wrapper.Mem
{
    public class MemHalf : Mem<Half>
    {
        public MemHalf(
            CommandQueue commandQueue,
            Context context,
            long arrayLength,
            MemFlags flags)
            : base(commandQueue, context, arrayLength, 2, flags)
        {
        }
    }
}