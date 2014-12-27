using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.Transposer
{
    public interface IOpenCLTransposer : ITransposer
    {
        MemFloat Destination
        {
            get;
        }
    }
}