namespace MyNN.MLP.DropConnect.WeightMask
{
    /// <summary>
    /// Weight mask container with Bernoulli mask.
    /// For details refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    public interface IWeightMaskContainer
    {
        void RegenerateMask();
    }
}