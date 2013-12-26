using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.Mem
{
    public class MemUint: Mem<uint>
    {
        public MemUint(
            Cl.CommandQueue commandQueue,
            Cl.Context context,
            int arrayLength,
            Cl.MemFlags flags)
            : base(commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
