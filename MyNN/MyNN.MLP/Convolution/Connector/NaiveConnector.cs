using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.Connector
{
    public interface IConnector
    {
        List<int> GetPreviousFeatureMapIndexes(
            int currentFmi
            );
    }

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
