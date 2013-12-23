using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.MLP2.Structure
{
    /// <summary>
    /// состояние слоя после просчета
    /// </summary>
    public interface ILayerState : IEnumerable<float>
    {
        float[] State
        {
            get;
        }
    }
}
