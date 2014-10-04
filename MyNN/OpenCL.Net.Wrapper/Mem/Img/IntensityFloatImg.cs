using System;

namespace OpenCL.Net.Wrapper.Mem.Img
{
    public class IntensityFloatImg
        : FloatImg
    {
        public IntensityFloatImg(
            Action<Guid> memDisposed, 
            CommandQueue commandQueue, 
            Context context, 
            uint width, 
            uint height, 
            MemFlags flags)
            : base(memDisposed, commandQueue, context, width, height, 1, ChannelOrder.Intensity, flags)
        {
        }
    }
}