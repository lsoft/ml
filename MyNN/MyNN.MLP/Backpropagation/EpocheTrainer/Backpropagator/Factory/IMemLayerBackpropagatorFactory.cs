using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator.Factory
{
    public interface IMemLayerBackpropagatorFactory
    {
        IMemLayerBackpropagator CreateOutputMemLayerBackpropagator(
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IKernelTextProvider kernelTextProvider,
            IMemDesiredValuesContainer desiredValuesContainer
            );

        IMemLayerBackpropagator CreateHiddenMemLayerBackpropagator(
            int layerIndex,
            IMemLayerContainer previousLayerContainer,
            IMemLayerContainer currentLayerContainer,
            IMemLayerContainer nextLayerContainer,
            IKernelTextProvider kernelTextProvider,
            MemFloat nextLayerDeDz
            );

    }
}