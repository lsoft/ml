

using System;

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemUint: Mem<uint>
    {
        public MemUint(
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
