using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.ArtifactContainer;

namespace MyNN.MLP.Backpropagation.Validation.Drawer.Factory
{
    public interface IDrawerFactory
    {
        IDrawer CreateDrawer(
            IArtifactContainer containerForSave,
            int? epocheNumber
            );
    }
}
