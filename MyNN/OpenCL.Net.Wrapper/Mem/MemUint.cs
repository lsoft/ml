

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemUint: Mem<uint>
    {
        public MemUint(
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
