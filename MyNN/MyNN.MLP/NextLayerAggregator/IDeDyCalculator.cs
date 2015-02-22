using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.NextLayerAggregator
{
    public interface IDeDyCalculator
    {
        void Aggregate(
            );

        void ClearAndWrite();
    }

    public interface ICSharpDeDyCalculator : IDeDyCalculator
    {
        float[] DeDy
        {
            get;
        }
    }

    public interface IOpenCLDeDyCalculator : IDeDyCalculator
    {
        MemFloat DeDy
        {
            get;
        }
    }
}