using OpenCL.Net.OpenCL.Mem;

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