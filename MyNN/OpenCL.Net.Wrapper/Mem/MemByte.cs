using System;

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemByte : Mem<byte>
    {
        public MemByte(
            Action<Guid> memDisposed,
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(memDisposed, commandQueue, context, arrayLength, 1, flags)
        {
        }
    }
}