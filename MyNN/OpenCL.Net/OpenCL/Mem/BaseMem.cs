using System;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.Mem
{
    public abstract class BaseMem : IDisposable
    {
        protected Cl.Mem _mem;

        public Cl.Mem GetMem()
        {
            return _mem;
        }

        public virtual void Dispose()
        {
            Cl.ReleaseMemObject(_mem);
        }
    }
}