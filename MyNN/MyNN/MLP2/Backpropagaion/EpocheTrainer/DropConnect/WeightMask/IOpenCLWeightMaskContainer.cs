using OpenCL.Net.OpenCL.Mem;

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