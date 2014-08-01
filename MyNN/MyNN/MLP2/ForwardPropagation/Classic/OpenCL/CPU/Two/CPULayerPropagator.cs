using System;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two
{
    public class CPULayerPropagator : ICPULayerPropagator
    {
        private readonly ILayerMemContainer _previousLayerMemContainer;
        private readonly ILayerMemContainer _currentLayerMemContainer;
        private readonly int _prevLayerNeuronTotalCount;
        private readonly int _currentLayerNonBiasNeuronCount;
        
        private readonly Kernel _kernel;

        public CPULayerPropagator(
            CLProvider clProvider,
            CPUKernelSource ks,
            ILayerMemContainer previousLayerMemContainer,
            ILayerMemContainer currentLayerMemContainer,
            IFunction activationFunction,
            VectorizationSizeEnum vse,
            int prevLayerNeuronTotalCount,
            int currentLayerNonBiasNeuronCount
            )
        {
            if (ks == null)
            {
                throw new ArgumentNullException("ks");
            }
            if (previousLayerMemContainer == null)
            {
                throw new ArgumentNullException("previousLayerMemContainer");
            }
            if (currentLayerMemContainer == null)
            {
                throw new ArgumentNullException("currentLayerMemContainer");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            _previousLayerMemContainer = previousLayerMemContainer;
            _currentLayerMemContainer = currentLayerMemContainer;
            _prevLayerNeuronTotalCount = prevLayerNeuronTotalCount;
            _currentLayerNonBiasNeuronCount = currentLayerNonBiasNeuronCount;

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
            _kernel
                .SetKernelArgMem(0, _previousLayerMemContainer.StateMem)
                .SetKernelArgMem(1, _currentLayerMemContainer.NetMem)
                .SetKernelArgMem(2, _currentLayerMemContainer.StateMem)
                .SetKernelArgMem(3, _currentLayerMemContainer.WeightMem)
                .SetKernelArg(4, 4, _prevLayerNeuronTotalCount)
                .EnqueueNDRangeKernel(_currentLayerNonBiasNeuronCount);
        }
    }
}
