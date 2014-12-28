namespace MyNN.MLP.DesiredValues
{
    public interface ICSharpDesiredValuesContainer : IDesiredValuesContainer
    {
        float[] DesiredOutput
        {
            get;
        }

    }
}