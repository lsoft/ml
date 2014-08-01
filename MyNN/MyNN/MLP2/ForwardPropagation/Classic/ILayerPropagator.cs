namespace MyNN.MLP2.ForwardPropagation.Classic
{
    public interface ILayerPropagator
    {
        void ComputeLayer(
            );

        void WaitForCalculationFinished();
    }
}