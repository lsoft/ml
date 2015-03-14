namespace MyNN.MLP.Structure.Layer
{
    public interface IAvgPoolingLayerConfiguration : ILayerConfiguration
    {
        int FeatureMapCount
        {
            get;
        }

        int InverseScaleFactor
        {
            get;
        }

        float ScaleFactor
        {
            get;
        }
    }
}