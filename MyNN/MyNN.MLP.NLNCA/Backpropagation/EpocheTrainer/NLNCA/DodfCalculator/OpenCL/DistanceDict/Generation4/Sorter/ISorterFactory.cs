using OpenCL.Net.Wrapper;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4.Sorter
{
    public interface ISorterFactory
    {
        ISorter CreateSorter(CLProvider clProvider);
    }
}