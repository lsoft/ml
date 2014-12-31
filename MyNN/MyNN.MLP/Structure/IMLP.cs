using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure
{
    public interface IMLP
    {
        string Name
        {
            get;
        }

        ILayer[] Layers
        {
            get;
        }

        string GetLayerInformation();

        void AutoencoderCutTail();

        void CutLastLayer();

        void AutoencoderCutHead();

        void AddLayer(
            IFunction activationFunction,
            int TotalNeuronCount,
            bool isNeedBiasNeuron);

        IMLPConfiguration GetConfiguration();

    }
}