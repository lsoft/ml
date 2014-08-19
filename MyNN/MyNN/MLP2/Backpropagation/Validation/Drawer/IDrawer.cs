using System.Collections.Generic;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Backpropagation.Validation.Drawer
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