namespace MyNN.MLP.Structure.Layer
{
    public interface IDimension
    {
        int DimensionCount
        {
            get;
        }

        int[] Sizes
        {
            get;
        }

        int TotalNeuronCount
        {
            get;
        }

        string GetDimensionInformation(
            );
    }
}