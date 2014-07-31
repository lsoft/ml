using System;

namespace OpenCL.Net.Wrapper.Mem
{
    public interface IMemWrapper : IDisposable
    {
        Guid MemGuid
        {
            get;
        }
    }
}