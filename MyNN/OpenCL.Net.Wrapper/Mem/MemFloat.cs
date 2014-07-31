

using System;

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemFloat : Mem<float>
    {
        public MemFloat(
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
