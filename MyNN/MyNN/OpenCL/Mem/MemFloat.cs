using OpenCL.Net;

namespace MyNN.OpenCL.Mem
{
    public class MemFloat : Mem<float>
    {
        public MemFloat(
            Cl.CommandQueue commandQueue,
            Cl.Context context,
            int arrayLength,
            Cl.MemFlags flags)
            : base(commandQueue, context, arrayLength, 4, flags)
        {
        }
    }
}
