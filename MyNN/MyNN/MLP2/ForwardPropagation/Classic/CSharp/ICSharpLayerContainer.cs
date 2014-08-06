﻿using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp
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