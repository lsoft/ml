using System;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DeDyAggregator
{
    /// <summary>
    ///  онтейнер, который используетс€ на слое свертки, если на данном слое не надо
    /// аггрегировать dedy дл€ предыдущего сло€.
    /// “акое может быть, если текущий слой - первый скрытый слой.
    /// </summary>
    public class CSharpStubConvolutionDeDyAggregator : ICSharpDeDyAggregator
    {
        private readonly IConvolutionLayerConfiguration _aggregateLayerConfiguration;

        public float[] DeDz
        {
            get;
            private set;
        }

        public float[] DeDy
        {
            get;
            private set;
        }

        public int TotalNeuronCount
        {
            get
            {
                return
                    _aggregateLayerConfiguration.TotalNeuronCount;
            }
        }

        public CSharpStubConvolutionDeDyAggregator(
            IConvolutionLayerConfiguration aggregateLayerConfiguration
            )
        {
            if (aggregateLayerConfiguration == null)
            {
                throw new ArgumentNullException("aggregateLayerConfiguration");
            }

            _aggregateLayerConfiguration = aggregateLayerConfiguration;

            this.DeDz = new float[aggregateLayerConfiguration.TotalNeuronCount];
            this.DeDy = new float[0];
        }

        public void Aggregate(
            )
        {
            //nothing to do
        }

        public void ClearAndWrite(
            )
        {
            this.DeDy.Clear();
            this.DeDz.Clear();
        }

    }
}