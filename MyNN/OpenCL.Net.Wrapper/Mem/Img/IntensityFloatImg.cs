using System;

namespace OpenCL.Net.Wrapper.Mem.Img
{
    public class IntensityFloatImg
        : IMemWrapper
    {
        private readonly Action<Guid> _memDisposed;
        private readonly CommandQueue _commandQueue;
        private readonly uint _width;
        private readonly uint _height;

        private readonly IMem _mem;
        
        public float[] Array;

        public Guid MemGuid
        {
            get;
            private set;
        }

        public IntensityFloatImg(
            Action<Guid> memDisposed,
            CommandQueue commandQueue,
            Context context,
            uint width,
            uint height,
            MemFlags flags)
        {
            if (memDisposed == null)
            {
                throw new ArgumentNullException("memDisposed");
            }
            _memDisposed = memDisposed;
            _commandQueue = commandQueue;
            _width = width;
            _height = height;

            Array = new float[width * height];

            MemGuid = Guid.NewGuid();


            ErrorCode errorcodeRet;
            _mem = Cl.CreateImage2D(
                context,
                flags,
                new ImageFormat(
                    ChannelOrder.Intensity,
                    ChannelType.Float),
                (IntPtr)width,
                (IntPtr)height,
                (IntPtr)0,
                Array,
                out errorcodeRet);
        }

        public IMem GetMem()
        {
            return _mem;
        }

        public void Write(
            BlockModeEnum blockMode)
        {
            var blocking = blockMode == BlockModeEnum.Blocking ? Bool.True : Bool.False;

            var originPtr = new IntPtr[] { (IntPtr)0, (IntPtr)0, (IntPtr)0 };    //x, y, z
            var regionPtr = new IntPtr[] { (IntPtr)_width, (IntPtr)_height, (IntPtr)1 };    //x, y, z

            Event writeEvent;

            var error = Cl.EnqueueWriteImage(
                _commandQueue,
                _mem,
                blocking,
                originPtr,
                regionPtr,
                (IntPtr) 0,
                (IntPtr) 0,
                Array,
                0,
                null,
                out writeEvent
                );


            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("EnqueueWriteImage failed: {0}!", error));
            }

            writeEvent.Dispose();
        }

        public void Read(BlockModeEnum blockMode)
        {
            var blocking = blockMode == BlockModeEnum.Blocking ? Bool.True : Bool.False;

            var originPtr = new IntPtr[] { (IntPtr)0, (IntPtr)0, (IntPtr)0 };    //x, y, z
            var regionPtr = new IntPtr[] { (IntPtr)_width, (IntPtr)_height, (IntPtr)1 };    //x, y, z

            Event writeEvent;
            var error = Cl.EnqueueReadImage(
                _commandQueue,
                _mem, 
                blocking,
                originPtr,
                regionPtr,
                (IntPtr)0,
                (IntPtr)0,
                Array,
                0, 
                null, 
                out writeEvent
                );

            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("EnqueueReadImage failed: {0}!", error));
            }

            writeEvent.Dispose();
        }
        public void Dispose()
        {
            this.Array = null;

            Cl.ReleaseMemObject(_mem);

            _memDisposed(this.MemGuid);
        }
    }
}