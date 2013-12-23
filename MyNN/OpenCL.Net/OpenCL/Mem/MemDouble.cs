using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.Mem
{
    public class MemDouble : Mem<double>
    {
        public MemDouble(
            Cl.CommandQueue commandQueue,
            Cl.Context context,
            int arrayLength,
            Cl.MemFlags flags)
            : base(commandQueue, context, arrayLength, 8, flags)
        {
        }

    }
}
