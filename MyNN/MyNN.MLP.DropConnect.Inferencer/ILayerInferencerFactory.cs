using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DropConnect.Inferencer
{
    public interface ILayerInferencerFactory
    {
        ILayerInferencer CreateLayerInferencer(
            ILayer previousLayer,
            ILayer currentLayer,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer
            );
    }
}