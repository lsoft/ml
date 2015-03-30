using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.Connector
{
    public class FullConnectedConnector : IConnector
    {
        private readonly IAvgPoolingLayerConfiguration _previousLayerConfiguration;

        public FullConnectedConnector(
            IAvgPoolingLayerConfiguration previousLayerConfiguration
            )
        {
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }
            _previousLayerConfiguration = previousLayerConfiguration;
        }

        public List<int> GetPreviousFeatureMapIndexes(int currentFmi)
        {
            var result = Enumerable
                .Range(0, _previousLayerConfiguration.FeatureMapCount)
                .ToList();

            return 
                result;
        }
    }
}