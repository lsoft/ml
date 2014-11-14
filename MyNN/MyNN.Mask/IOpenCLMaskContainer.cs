using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.Mask
{
    /// <summary>
    /// Weight mask container with bit Bernoulli mask.
    /// </summary>
    public interface IOpenCLMaskContainer : IMaskContainer
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