using System.Drawing;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Function;


namespace MyNN.MLP2.Structure
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
            int nonBiasNeuronCount,
            bool isNeedBiasNeuron);


    }
}