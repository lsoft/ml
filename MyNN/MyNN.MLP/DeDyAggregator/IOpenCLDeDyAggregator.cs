using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.DeDyAggregator
{
    public interface IOpenCLDeDyAggregator : IDeDyAggregator
    {
        MemFloat DeDz
        {
            get;
        }

        MemFloat DeDy
        {
            get;
        }
    }
}