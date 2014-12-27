using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.DesiredValues
{
    public interface IMemDesiredValuesContainer : IDesiredValuesContainer
    {
        MemFloat DesiredOutput
        {
            get;
        }
    }
}