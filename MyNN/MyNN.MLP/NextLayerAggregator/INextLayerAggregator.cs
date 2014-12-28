using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.NextLayerAggregator
{
    public interface INextLayerAggregator
    {
        MemFloat PreprocessCache
        {
            get;
        }

        void Aggregate(
            );

        void ClearAndWrite();
    }
}