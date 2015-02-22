using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.DeDyAggregator
{
    public interface IOpenCLDeDyAggregator : IDeDyAggregator
    {
        MemFloat DeDy
        {
            get;
        }
    }
}