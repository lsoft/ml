using System.Collections.Generic;

namespace MyNN.MLP.Structure.Layer
{
    /// <summary>
    /// состояние слоя после просчета
    /// </summary>
    public interface ILayerState : IEnumerable<float>
    {
        float[] NState
        {
            get;
        }
    }
}
