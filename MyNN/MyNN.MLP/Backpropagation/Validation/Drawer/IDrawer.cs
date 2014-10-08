using System.Collections.Generic;
using MyNN.Common.ArtifactContainer;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.Drawer
{
    public interface IDrawer
    {
        void Draw(
            IArtifactContainer containerForSave,
            int? epocheNumber,
            List<ILayerState> netResults
            );
    }
}