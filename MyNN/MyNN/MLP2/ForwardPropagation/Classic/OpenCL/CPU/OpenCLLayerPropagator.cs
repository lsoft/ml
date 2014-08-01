using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU
{
    public class OpenCLLayerPropagator
    {
        private readonly Kernel _kernel;

        public OpenCLLayerPropagator(
            CLProvider clProvider,
            CPUKernelSource ks,
            IFunction activationFunction,
            VectorizationSizeEnum vse
            )
        {
            if (ks == null)
            {
                throw new ArgumentNullException("ks");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            string kernelName;
            var kernelSource = ks.GetKernelSource(
                vse,
                activationFunction,
                out kernelName
                );

            _kernel = clProvider.CreateKernel(
                kernelSource,
                kernelName);
        }

        public void ComputeLayer(
            )
        {
            //var prevLayerNeuronTotalCount = _mlp.Layers[layerIndex - 1].Neurons.Length;

            //var vectorizationSize = VectorizationHelper.GetVectorizationSize(_vse);

            //_kernel
            //    .SetKernelArgMem(0, this.StateMem[layerIndex - 1])
            //    .SetKernelArgMem(1, this.NetMem[layerIndex])
            //    .SetKernelArgMem(2, this.StateMem[layerIndex])
            //    .SetKernelArgMem(3, this.WeightMem[layerIndex])
            //    .SetKernelArg(4, 4, prevLayerNeuronTotalCount / vectorizationSize)
            //    .SetKernelArg(5, 4, prevLayerNeuronTotalCount - prevLayerNeuronTotalCount % vectorizationSize)
            //    .SetKernelArg(6, 4, prevLayerNeuronTotalCount)
            //    .EnqueueNDRangeKernel(_mlp.Layers[layerIndex].NonBiasNeuronCount);
        }
    }
}
