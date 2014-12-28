using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel
{
    internal class UpdateWeightKernel
    {
        public void UpdateWeigths(
            float[] currentLayerWeights,
            float[] nabla,
            float batchSize
            )
        {
            if (currentLayerWeights == null)
            {
                throw new ArgumentNullException("currentLayerWeights");
            }
            if (nabla == null)
            {
                throw new ArgumentNullException("nabla");
            }

            Parallel.For(0, currentLayerWeights.Length, cc =>
            //for (var cc = 0; cc < currentLayerWeights.Length; cc++)
            {
                currentLayerWeights[cc] += nabla[cc] / batchSize;
            }
            ); //Parallel.For

        }
    }
}
