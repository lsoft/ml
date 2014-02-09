

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemDouble : Mem<double>
    {
        public MemDouble(
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(commandQueue, context, arrayLength, 8, flags)
        {
        }

    }
}
