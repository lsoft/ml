using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask
{
    /// <summary>
    /// Weight mask container with bit Bernoulli mask.
    /// </summary>
    public interface IOpenCLWeightBitMaskContainer : IWeightMaskContainer
    {
        /// <summary>
        /// Number of bit, used in current iteration ([0;31])
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