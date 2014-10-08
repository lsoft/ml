using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.Tests.MLP2
{
    public class MLPConfigurationConstructor
    {
        public static IMLPConfiguration CreateConfiguration(
            int[] layerSizes
            )
        {
            if (layerSizes == null)
            {
                throw new ArgumentNullException("layerSizes");
            }

            var lcl = new List<ILayerConfiguration>();
            for (var li = 0; li < layerSizes.Length; li++)
            {
                var firstLayer = li == 0;
                var lastLayer = li == layerSizes.Length - 1;

                var prevLayerNonBiasNeuron = firstLayer ? 0 : layerSizes[li - 1] + 1;
                var prevLayerTotalNeuron = firstLayer ? 0 : layerSizes[li - 1] + 1;
                var currentLauerTotalNeuron = layerSizes[li] + (lastLayer ? 0 : 1);

                var ncl = new List<INeuronConfiguration>();
                for (var ni = 0; ni < currentLauerTotalNeuron; ni++)
                {
                    var isBiasNeuron = !lastLayer && ni == (currentLauerTotalNeuron - 1);
                    var weightsCount = isBiasNeuron ? 0 : prevLayerTotalNeuron;

                    var nc = new NeuronConfiguration(
                        weightsCount,
                        isBiasNeuron);

                    ncl.Add(nc);
                }

                var lc = new LayerConfiguration(
                    ncl.ToArray(),
                    !lastLayer,
                    layerSizes[li]
                    );

                lcl.Add(lc);
            }

            var result = new MLPConfiguration(
                lcl.ToArray());

            return result;
        }
    }
}
