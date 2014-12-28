namespace MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator
{
    public interface ICSharpLayerBackpropagator : ILayerBackpropagator
    {
        float[] DeDz
        {
            get;
        }
    }
}