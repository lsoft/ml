using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp.Two
{
    public interface ICSharpLayerContainer : ILayerContainer
    {
        float[] WeightMem
        {
            get;
        }

        float[] NetMem
        {
            get;
        }

        float[] StateMem
        {
            get;
        }
    }
}