using System.Collections.Generic;
using MyNN.MLP2.Container;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer
{
    public interface IDrawer
    {
        void Draw(
            IMLPContainer containerForSave,
            int? epocheNumber,
            List<ILayerState> netResults
            );
    }
}