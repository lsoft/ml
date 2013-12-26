using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnect.WeightMask;
using OpenCL.Net.OpenCL.Mem;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit.WeightMask
{
    public interface IOpenCLWeightBitMaskContainer : IWeightMaskContainer
    {
        uint BitMask
        {
            get;
        }

        MemUint[] MaskMem
        {
            get;
        }
    }
}