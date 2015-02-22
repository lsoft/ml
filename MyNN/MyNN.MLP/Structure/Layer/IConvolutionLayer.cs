namespace MyNN.MLP.Structure.Layer
{
    public interface IConvolutionLayer : ILayer
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