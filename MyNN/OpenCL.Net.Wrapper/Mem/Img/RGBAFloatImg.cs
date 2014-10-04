using System;

namespace OpenCL.Net.Wrapper.Mem.Img
{
    public class RGBAFloatImg
        : FloatImg
    {
        public RGBAFloatImg(
            Action<Guid> memDisposed,
            CommandQueue commandQueue,
            Context context,
            uint width,
            uint height,
            MemFlags flags)
            : base(memDisposed, commandQueue, context, width, height, 4, ChannelOrder.RGBA, flags)
        {
        }
    }
}