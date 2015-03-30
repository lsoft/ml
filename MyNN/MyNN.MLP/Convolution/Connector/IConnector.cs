using System.Collections.Generic;

namespace MyNN.MLP.Convolution.Connector
{
    public interface IConnector
    {
        List<int> GetPreviousFeatureMapIndexes(
            int currentFmi
            );
    }
}