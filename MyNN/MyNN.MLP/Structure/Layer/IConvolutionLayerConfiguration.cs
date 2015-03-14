namespace MyNN.MLP.Structure.Layer
{
    public interface IConvolutionLayerConfiguration : ILayerConfiguration
    {
        IDimension KernelSpatialDimension
        {
            get;
        }

        int FeatureMapCount
        {
            get;
        }
    }
}