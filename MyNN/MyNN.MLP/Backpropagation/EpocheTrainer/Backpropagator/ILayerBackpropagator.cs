namespace MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator
{
    public interface ILayerBackpropagator
    {
        void Prepare(
            );

        void Backpropagate(
            int dataCount,
            float learningRate,
            bool firstItemInBatch
            );

        void UpdateWeights(
            );
    }
}