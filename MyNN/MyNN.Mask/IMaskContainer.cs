namespace MyNN.Mask
{
    /// <summary>
    /// Weight mask container with Bernoulli mask.
    /// For details refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    public interface IMaskContainer
    {
        void RegenerateMask();
    }
}