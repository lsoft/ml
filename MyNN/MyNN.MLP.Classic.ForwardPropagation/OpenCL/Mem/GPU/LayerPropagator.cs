using System;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU
{
    public class LayerPropagator : ILayerPropagator
    {
        private readonly CLProvider _clProvider;
        private readonly IMemLayerContainer _previousMemLayerContainer;
        private readonly IMemLayerContainer _currentMemLayerContainer;

        private readonly Kernel _kernel;

        public LayerPropagator(
            CLProvider clProvider,
            IMemLayerContainer previousMemLayerContainer,
            IMemLayerContainer currentMemLayerContainer,
            IFunction activationFunction
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (previousMemLayerContainer == null)
            {
                throw new ArgumentNullException("previousMemLayerContainer");
            }
            if (currentMemLayerContainer == null)
            {
                throw new ArgumentNullException("currentMemLayerContainer");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            _clProvider = clProvider;
            _previousMemLayerContainer = previousMemLayerContainer;
            _currentMemLayerContainer = currentMemLayerContainer;

            var ks = new MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU.KernelSource();

            string kernelName;
            var kernelSource = ks.GetKernelSource(
                activationFunction,
                currentMemLayerContainer.Configuration.TotalNeuronCount,
                previousMemLayerContainer.Configuration.TotalNeuronCount,
                out kernelName
                );

            _kernel = clProvider.CreateKernel(
                kernelSource,
                kernelName);
        }

        public void ComputeLayer(
            )
        {
            const uint szLocalWorkSize = 256;
            uint szGlobalWorkSize = 64 * _clProvider.Parameters.NumComputeUnits * szLocalWorkSize;

            _kernel
                .SetKernelArgMem(0, _previousMemLayerContainer.StateMem)
                .SetKernelArgMem(1, _currentMemLayerContainer.NetMem)
                .SetKernelArgMem(2, _currentMemLayerContainer.StateMem)
                .SetKernelArgMem(3, _currentMemLayerContainer.WeightMem)
                .SetKernelArgLocalMem(4, 4 * szLocalWorkSize)
                .SetKernelArgMem(5, _currentMemLayerContainer.BiasMem)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        szGlobalWorkSize
                    }
                    , new[]
                    {
                        szLocalWorkSize
                    }
                    );
        }

        public void WaitForCalculationFinished()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }
    }
}
