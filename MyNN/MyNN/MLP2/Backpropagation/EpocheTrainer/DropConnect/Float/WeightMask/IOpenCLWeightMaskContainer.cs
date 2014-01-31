using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.WeightMask
{
    /// <summary>
    /// Weight mask container with float Bernoulli mask.
    /// </summary>
    public interface IOpenCLWeightMaskContainer : IWeightMaskContainer
    {
        /// <summary>
        /// OpenCL buffer for mask
        /// </summary>
        MemFloat[] MaskMem
        {
            get;
        }
    }
}