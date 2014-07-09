using System.Drawing;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Function;


namespace MyNN.MLP2.Structure
{
    public interface IMLP
    {
        string Root
        {
            get;
        }

        string FolderName
        {
            get;
        }

        string WorkFolderPath
        {
            get;
        }

        ILayer[] Layers
        {
            get;
        }

        void SetRootFolder(string root);

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