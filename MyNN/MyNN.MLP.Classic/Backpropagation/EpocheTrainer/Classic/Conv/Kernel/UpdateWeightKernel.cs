using System;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel
{
    internal class UpdateWeightKernel
    {
        public void UpdateWeigths(
            IReferencedKernelBiasContainer kbContainer,
            IReferencedSquareFloat nablaContainer,
            float nablaBias,
            float batchSize
            )
        {
            if (kbContainer == null)
            {
                throw new ArgumentNullException("kbContainer");
            }
            if (nablaContainer == null)
            {
                throw new ArgumentNullException("nablaContainer");
            }

            kbContainer.IncrementBy(
                nablaContainer,
                nablaBias,
                batchSize
                );
        }
    }
}
