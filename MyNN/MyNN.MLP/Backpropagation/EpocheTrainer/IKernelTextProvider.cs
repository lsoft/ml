using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.EpocheTrainer
{
    public interface IKernelTextProvider
    {
        string GetOverwriteCalculationKernelsSource(
            ILayerConfiguration layerConfiguration
            );
        
        string GetIncrementCalculationKernelsSource(
            ILayerConfiguration layerConfiguration
            );

        string UpdateWeightKernelSource
        {
            get;
        }
    }
}