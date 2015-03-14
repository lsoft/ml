using System;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public class CSharpConvolutionLayerContainer : CSharpLayerContainer, ICSharpConvolutionLayerContainer
    {
        public new IConvolutionLayerConfiguration Configuration
        {
            get;
            private set;
        }


        public CSharpConvolutionLayerContainer(IConvolutionLayerConfiguration layerConfiguration)
            : base(layerConfiguration)
        {
            Configuration = layerConfiguration;
        }
    }

    //public class CSharpConvolutionLayerContainer : ICSharpConvolutionLayerContainer
    //{
    //    public IConvolutionLayerConfiguration Configuration
    //    {
    //        get;
    //        private set;
    //    }

    //    ILayerConfiguration ILayerContainer.Configuration
    //    {
    //        get
    //        {
    //            return
    //                this.Configuration;
    //        }
    //    }

    //    public float[] WeightMem
    //    {
    //        get;
    //        private set;
    //    }

    //    public float[] BiasMem
    //    {
    //        get;
    //        private set;
    //    }

    //    public float[] NetMem
    //    {
    //        get;
    //        private set;
    //    }

    //    public float[] StateMem
    //    {
    //        get;
    //        private set;
    //    }

    //    public CSharpConvolutionLayerContainer(
    //        IConvolutionLayerConfiguration layerConfiguration
    //        )
    //    {
    //        if (layerConfiguration == null)
    //        {
    //            throw new ArgumentNullException("layerConfiguration");
    //        }

    //        Configuration = layerConfiguration;

    //        var totalNeuronCount = layerConfiguration.TotalNeuronCount;
    //        var weightCount = layerConfiguration.WeightCount;
    //        var biasCount = layerConfiguration.BiasCount;

    //        //веса
    //        WeightMem = weightCount > 0 ? new float[weightCount] : null;
    //        BiasMem = biasCount > 0 ? new float[biasCount] : null;

    //        //нейроны
    //        NetMem = new float[totalNeuronCount];
    //        StateMem = new float[totalNeuronCount];
    //    }


    //    public void ClearAndPushNetAndState()
    //    {
    //        ClearNetAndState();

    //        PushNetAndState();
    //    }

    //    public void ClearNetAndState()
    //    {
    //        var nml = this.NetMem.Length;
    //        Array.Clear(this.NetMem, 0, nml);

    //        var sml = this.StateMem.Length;
    //        Array.Clear(this.StateMem, 0, sml);
    //    }

    //    public void PushNetAndState()
    //    {
    //        //nothing to do
    //    }

    //    public void ReadInput(float[] data)
    //    {
    //        if (data == null)
    //        {
    //            throw new ArgumentNullException("data");
    //        }
    //        if (data.Length != Configuration.TotalNeuronCount)
    //        {
    //            throw new ArgumentException("data.Length != Configuration.TotalNeuronCount");
    //        }

    //        //записываем значения из сети в объекты OpenCL
    //        for (var neuronIndex = 0; neuronIndex < Configuration.TotalNeuronCount; neuronIndex++)
    //        {
    //            this.NetMem[neuronIndex] = 0; //LastNET
    //            this.StateMem[neuronIndex] = data[neuronIndex];
    //        }
    //    }

    //    public void ReadWeightsAndBiasesFromLayer(ILayer layer)
    //    {
    //        if (layer == null)
    //        {
    //            throw new ArgumentNullException("layer");
    //        }

    //        if (this.WeightMem != null || this.BiasMem != null)
    //        {
    //            float[] weightMem;
    //            float[] biasMem;
    //            layer.GetClonedWeights(
    //                out weightMem,
    //                out biasMem
    //                );

    //            if (this.WeightMem != null)
    //            {
    //                weightMem.CopyTo(this.WeightMem, 0);
    //            }

    //            if (this.BiasMem != null)
    //            {
    //                biasMem.CopyTo(this.BiasMem, 0);
    //            }
    //        }
    //    }

    //    public void PopNetAndState()
    //    {
    //        //nothing to do
    //    }

    //    public void PopWeightsAndBiases()
    //    {
    //        //nothing to do
    //    }

    //    public void WritebackWeightsAndBiasesToMLP(ILayer layer)
    //    {
    //        if (this.WeightMem != null && this.BiasMem != null)
    //        {
    //            layer.SetWeights(
    //                this.WeightMem,
    //                this.BiasMem
    //                );
    //        }
    //    }

    //    public ILayerState GetLayerState()
    //    {
    //        var ls = new LayerState(
    //            this.StateMem.CloneArray(),
    //            Configuration.TotalNeuronCount
    //            );

    //        return ls;
    //    }
    //}
}