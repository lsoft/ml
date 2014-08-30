namespace MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL
{
    internal interface IKernelTextProvider
    {
        string GetOverwriteCalculationKernelsSource(int layerIndex);
        
        string GetIncrementCalculationKernelsSource(int layerIndex);

        string UpdateWeightKernelSource
        {
            get;
        }
    }
}