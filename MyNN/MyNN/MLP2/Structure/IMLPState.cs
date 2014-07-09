using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Structure
{
    public interface IMLPState
    {
        ILayerState[] State
        {
            get;
        }
    }
}
