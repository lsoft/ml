using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnect.WeightMask
{
    public interface IOpenCLWeightMaskContainer : IWeightMaskContainer
    {
        MemFloat[] MaskMem
        {
            get;
        }
    }
}