

using System;
using OpenCL.Net.Extensions;

namespace OpenCL.Net.Wrapper.Mem
{
    public abstract class Mem<T>
        : IMemWrapper
        where T : struct
    {
        private readonly Action<Guid> _memDisposed;
        private readonly CommandQueue _commandQueue;
        private readonly IMem<T> _mem;

        public T[] Array;

        protected int _sizeOfT
        {
            get;
            private set;
        }

        public Guid MemGuid
        {
            get;
            private set;
        }

        protected Mem(
            Action<Guid> memDisposed,
            CommandQueue commandQueue,
            Context context,
            ulong arrayLength,
            int sizeOfT,
            MemFlags flags)
        {
            if (memDisposed == null)
            {
                throw new ArgumentNullException("memDisposed");
            }
            
            _memDisposed = memDisposed;
            _commandQueue = commandQueue;
            _sizeOfT = sizeOfT;

            Array = new T[arrayLength];

            MemGuid = Guid.NewGuid();

            _mem = context.CreateBuffer(Array, flags);
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

            _memDisposed(this.MemGuid);
        }
    }
}