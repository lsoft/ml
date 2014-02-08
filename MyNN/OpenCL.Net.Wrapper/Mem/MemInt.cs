﻿

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemInt: Mem<int>
    {
        public MemInt(
            CommandQueue commandQueue,
            Context context,
            long arrayLength,
            MemFlags flags)
            : base(commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
