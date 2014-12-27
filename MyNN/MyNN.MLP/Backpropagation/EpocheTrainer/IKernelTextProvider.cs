namespace MyNN.MLP.Backpropagation.EpocheTrainer
{
    public interface IKernelTextProvider
    {
        string GetPreprocessHiddenKernelZeroSource(
            int groupSize
            );

        string GetPreprocessHiddenKernelOneSource(
            );

        string GetOverwriteCalculationKernelsSource(int layerIndex);
        
        string GetIncrementCalculationKernelsSource(int layerIndex);

        string UpdateWeightKernelSource
        {
            get;
        }
    }
}