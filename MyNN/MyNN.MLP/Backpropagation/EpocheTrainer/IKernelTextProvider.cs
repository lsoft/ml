namespace MyNN.MLP.Backpropagation.EpocheTrainer
{
    public interface IKernelTextProvider
    {
        string GetOverwriteCalculationKernelsSource(int layerIndex);
        
        string GetIncrementCalculationKernelsSource(int layerIndex);

        string UpdateWeightKernelSource
        {
            get;
        }
    }
}