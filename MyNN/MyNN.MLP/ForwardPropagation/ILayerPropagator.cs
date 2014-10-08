namespace MyNN.MLP.ForwardPropagation
{
    public interface ILayerPropagator
    {
        void ComputeLayer(
            );

        void WaitForCalculationFinished();
    }
}