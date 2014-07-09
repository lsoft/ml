using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Structure.Layer.Factory.WeightLoader
{
    public interface IWeightLoader
    {
        void LoadWeights(ILayer layer);
    }
}
