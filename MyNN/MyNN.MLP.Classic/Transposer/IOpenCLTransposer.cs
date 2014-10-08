using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.Classic.Transposer
{
    public interface IOpenCLTransposer : ITransposer
    {
        MemFloat Destination
        {
            get;
        }
    }
}