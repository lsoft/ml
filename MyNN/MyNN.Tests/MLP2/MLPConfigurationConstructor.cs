using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2
{
    //public class MLPConfigurationConstructor
    //{
    //    public static IMLPConfiguration CreateConfiguration(
    //        IFunction layerActivationFunction,
    //        int[] layerSizes
    //        )
    //    {
    //        if (layerActivationFunction == null)
    //        {
    //            throw new ArgumentNullException("layerActivationFunction");
    //        }
    //        if (layerSizes == null)
    //        {
    //            throw new ArgumentNullException("layerSizes");
    //        }

    //        var lcl = new List<ILayerConfiguration>();
    //        for (var li = 0; li < layerSizes.Length; li++)
    //        {
    //            var firstLayer = li == 0;

    //            var prevLayerTotalNeuron = firstLayer ? 0 : layerSizes[li - 1];
    //            var currentLayerTotalNeuron = layerSizes[li];

    //            var ncl = new List<INeuronConfiguration>();
    //            for (var ni = 0; ni < currentLayerTotalNeuron; ni++)
    //            {
    //                var weightsCount = prevLayerTotalNeuron;

    //                var nc = new NeuronConfiguration(
    //                    weightsCount
    //                    );

    //                ncl.Add(nc);
    //            }

    //            var lc = new LayerConfiguration(
    //                layerActivationFunction,
    //                new Dimension(1, layerSizes[li]), 
    //                firstLayer ? 0 : prevLayerTotalNeuron * currentLayerTotalNeuron,
    //                firstLayer ? 0 : currentLayerTotalNeuron,
    //                ncl.ToArray()
    //                );

    //            lcl.Add(lc);
    //        }

    //        var result = new MLPConfiguration(
    //            lcl.ToArray());

    //        return result;
    //    }
    //}
}
