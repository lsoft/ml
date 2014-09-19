using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4.Sorter
{
    public interface ISorter
    {
        void Sort(
            MemByte dataMem,
            ulong totalElementCountPlusOverhead
            );
    }
}