using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.DropConnect.WeightMask
{
    /// <summary>
    /// Weight mask container with bit Bernoulli mask.
    /// </summary>
    public interface IOpenCLWeightMaskContainer : IWeightMaskContainer
    {
        /// <summary>
        /// Bit mask, used in current iteration (in sequence of 2^0, 2^1, 2^2, 2^3 ... to 2^32)
        /// </summary>
        uint BitMask
        {
            get;
        }

        /// <summary>
        /// OpenCL buffer for mask
        /// </summary>
        MemUint MaskMem
        {
            get;
        }
    }
}