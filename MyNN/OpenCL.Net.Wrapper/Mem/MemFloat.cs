

namespace OpenCL.Net.Wrapper.Mem
{
    public class MemFloat : Mem<float>
    {
        public MemFloat(
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
