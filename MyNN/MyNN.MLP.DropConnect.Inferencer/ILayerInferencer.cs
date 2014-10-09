namespace MyNN.MLP.DropConnect.Inferencer
{
    /// <summary>
    /// Stochastic layer inferencer
    /// For details refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    public interface ILayerInferencer
    {
        void InferenceLayer();
    }
}
