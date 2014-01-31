namespace MyNN.MLP2.ForwardPropagation.DropConnect.Inference
{
    /// <summary>
    /// Stochastic layer inferencer
    /// For details refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    public interface ILayerInference
    {
        void InferenceLayer();
    }
}
