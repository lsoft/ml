using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.MLP2.ForwardPropagation.DropConnect
{
    public interface ILayerInference
    {
        void InferenceLayer();
    }
}
