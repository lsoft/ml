using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.Structure.Layer.Factory.WeightLoader;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Layer.Factory
{
    public interface ILayerFactory
    {
        ILayer CreateInputLayer(int withoutBiasNeuronCount);

        ILayer CreateLayer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron
            );
    }
}
