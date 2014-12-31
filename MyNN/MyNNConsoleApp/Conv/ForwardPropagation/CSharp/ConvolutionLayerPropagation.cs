using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNNConsoleApp.Conv.ForwardPropagation.CSharp
{
    public class ConvolutionLayerPropagation : ILayerPropagator
    {
        public ConvolutionLayerPropagation(
            )
        {
        }

        public void ComputeLayer()
        {
            throw new NotImplementedException();
        }

        public void WaitForCalculationFinished()
        {
            //nothing to do
        }
    }


    public interface IConvolutionLayerContainer : ICSharpLayerContainer
    {
        float[] KernelMem
        {
            get;
        }
    }

    public class ConvolutionLayerContainer : IConvolutionLayerContainer
    {
        public float[] WeightMem
        {
            get;
            private set;
        }

        public float[] NetMem
        {
            get;
            private set;
        }

        public float[] StateMem
        {
            get;
            private set;
        }

        public float[] KernelMem
        {
            get;
            private set;
        }

        public ConvolutionLayerContainer(
            int convolutionWidth,
            int convolutionHeight,
            int kernelWidth,
            int kernelHeight
            )
        {
            this.NetMem = new float[convolutionWidth * convolutionHeight];
            this.StateMem = new float[convolutionWidth * convolutionHeight];

            this.KernelMem = new float[kernelWidth * kernelHeight];
        }

        public void ClearAndPushNetAndState()
        {
            throw new NotImplementedException();
        }

        public void ReadInput(float[] data)
        {
            throw new NotImplementedException();
        }

        public void ReadWeightsFromLayer(ILayer layer)
        {
            throw new NotImplementedException();
        }

        public void PopNetAndState()
        {
            throw new NotImplementedException();
        }

        public ILayerState GetLayerState()
        {
            throw new NotImplementedException();
        }

        public void PopWeights()
        {
            throw new NotImplementedException();
        }

        public void WritebackWeightsToMLP(ILayer layer)
        {
            throw new NotImplementedException();
        }
    }
}
