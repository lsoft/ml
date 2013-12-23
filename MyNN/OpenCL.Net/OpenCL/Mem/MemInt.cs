using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.Mem
{
    public class MemInt: Mem<int>
    {
        public MemInt(
            Cl.CommandQueue commandQueue,
            Cl.Context context,
            int arrayLength,
            Cl.MemFlags flags)
            : base(commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
