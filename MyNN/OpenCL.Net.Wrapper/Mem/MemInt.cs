

using System;

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemInt : Mem<int>
    {
        public MemInt(
            Action<Guid> memDisposed,
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(memDisposed, commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
