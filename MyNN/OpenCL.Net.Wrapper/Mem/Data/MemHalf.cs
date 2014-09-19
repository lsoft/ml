using System;

namespace OpenCL.Net.Wrapper.Mem.Data
{
    public class MemHalf : Mem<Half>
    {
        public MemHalf(
            Action<Guid> memDisposed,
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(memDisposed, commandQueue, context, arrayLength, 2, flags)
        {
        }
    }
}