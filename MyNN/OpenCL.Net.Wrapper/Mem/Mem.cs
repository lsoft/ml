

using System;
using OpenCL.Net.Extensions;

namespace OpenCL.Net.Wrapper.Mem
{
    public abstract class Mem<T> //: BaseMem
        : IMemWrapper
        where T : struct
    {
        private readonly CommandQueue _commandQueue;
        private readonly IMem<T> _mem;

        public T[] Array;

        protected int _sizeOfT
        {
            get;
            private set;
        }

        protected Mem(
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            int sizeOfT,
            MemFlags flags)
        {
            _commandQueue = commandQueue;
            _sizeOfT = sizeOfT;

            Array = new T[arrayLength];

            _mem = context.CreateBuffer(Array, flags);
            //_mem = context.AllocateArray<T>(Array, _sizeOfT, flags);
        }

        public OpenCL.Net.IMem<T> GetMem()
        {
            return _mem;
        }

        public void Write(BlockModeEnum blockMode)
        {
            var blocking = blockMode == BlockModeEnum.Blocking ? Bool.True : Bool.False;

            //var sizeInBytes = _sizeOfT * Array.Length;

            Event writeEvent;
            var error = Cl.EnqueueWriteBuffer(_commandQueue, _mem, blocking, Array, 0, null, out writeEvent);

            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("EnqueueWriteBuffer failed: {0}!", error));
            }

            writeEvent.Dispose();

            //_mem.Write(_commandQueue, _sizeOfT * Array.Length, Array, blockMode);
        }

        public void Read(BlockModeEnum blockMode)
        {
            var blocking = blockMode == BlockModeEnum.Blocking ? Bool.True : Bool.False;

            Event writeEvent;
            var error = Cl.EnqueueReadBuffer(_commandQueue, _mem, blocking, Array, 0, null, out writeEvent);

            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("EnqueueReadBuffer failed: {0}!", error));
            }

            writeEvent.Dispose();

            //_mem.Read(_commandQueue, _sizeOfT * Array.Length, Array, blockMode);
        }

        public virtual void Dispose()
        {
            this.Array = null;

            Cl.ReleaseMemObject(_mem);
        }
    }
}