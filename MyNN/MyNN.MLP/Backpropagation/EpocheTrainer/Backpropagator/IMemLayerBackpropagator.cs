using MyNN.MLP.DeDyAggregator;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator
{
    public interface IMemLayerBackpropagator : ILayerBackpropagator
    {
        MemFloat DeDz
        {
            get;
        }
    }

}