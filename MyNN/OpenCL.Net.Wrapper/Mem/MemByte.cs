namespace OpenCL.Net.Wrapper.Mem
{
    public class MemByte : Mem<byte>
    {
        public MemByte(
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            MemFlags flags)
            : base(commandQueue, context, arrayLength, 1, flags)
        {
        }
    }
}