using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.Connector
{
    public class NaiveConnector : IConnector
    {
        private readonly IAvgPoolingLayerConfiguration _previousLayerConfiguration;
        private readonly int _count;

        public NaiveConnector(
            IAvgPoolingLayerConfiguration previousLayerConfiguration,
            int count
            )
        {
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }

            _previousLayerConfiguration = previousLayerConfiguration;
            _count = count;
        }

        public List<int> GetPreviousFeatureMapIndexes(
            int currentFmi
            )
        {
            var result = new List<int>();

            for (var i = 0; i < _count; i++)
            {
                result.Add(
                    (currentFmi + i) % _previousLayerConfiguration.FeatureMapCount
                    );
            }

            return
                result;
        }
    }
}
