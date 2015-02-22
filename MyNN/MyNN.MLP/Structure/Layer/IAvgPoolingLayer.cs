namespace MyNN.MLP.Structure.Layer
{
    public interface IAvgPoolingLayer : ILayer
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