using OpenCL.Net.Wrapper.Mem;

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