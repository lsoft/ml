using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask
{
    /// <summary>
    /// Weight mask container with bit Bernoulli mask.
    /// </summary>
    public interface IOpenCLWeightBitMaskContainer : IWeightMaskContainer
    {
        /// <summary>
        /// Bit mask, used in current iteration (from 2^0 to 2^32)
        /// </summary>
        uint BitMask
        {
            get;
        }

        /// <summary>
        /// OpenCL buffer for mask
        /// </summary>
        MemUint[] MaskMem
        {
            get;
        }
    }
}