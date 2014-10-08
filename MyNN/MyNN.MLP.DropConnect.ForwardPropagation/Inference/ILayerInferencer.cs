using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DropConnect.ForwardPropagation.Inference
{
    /// <summary>
    /// Stochastic layer inferencer
    /// For details refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    public interface ILayerInferencer
    {
        void InferenceLayer();
    }

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
