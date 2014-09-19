using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP2.Transposer
{
    public interface IOpenCLTransposer : ITransposer
    {
        MemFloat Destination
        {
            get;
        }
    }
}