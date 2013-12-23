using OpenCL.Net.OpenCL.Mem;

namespace MyNN.MLP2.Transposer
{
    public interface IOpenCLTranposer : ITranposer
    {
        MemFloat Destination
        {
            get;
        }
    }
}