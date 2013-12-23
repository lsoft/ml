using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.Mem
{
    public abstract class Mem<T> : BaseMem
        where T : struct
    {
        private readonly Cl.CommandQueue _commandQueue;

        public T[] Array;

        protected int _sizeOfT
        {
            get;
            private set;
        }

        protected Mem(
            Cl.CommandQueue commandQueue,
            Cl.Context context,
            int arrayLength,
            int sizeOfT,
            Cl.MemFlags flags)
        {
            _commandQueue = commandQueue;
            _sizeOfT = sizeOfT;

            Array = new T[arrayLength];

            _mem = context.AllocateArray<T>(Array, _sizeOfT, flags);
        }

        public void Write(BlockModeEnum blockMode)
        {
            _mem.Write(_commandQueue, _sizeOfT * Array.Length, Array, blockMode);
        }

        public void Read(BlockModeEnum blockMode)
        {
            _mem.Read(_commandQueue, _sizeOfT * Array.Length, Array, blockMode);
        }

        public override void Dispose()
        {
            this.Array = null;

            base.Dispose();
        }
    }
}