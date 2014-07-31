

using System;

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemDouble : Mem<double>
    {
        public MemDouble(
            Action<Guid> memDisposed, 
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(memDisposed, commandQueue, context, arrayLength, 8, flags)
        {
        }

    }
}
