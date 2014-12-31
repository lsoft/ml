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
            float[] nablaWeights,
            float batchSize,
            float[] currentLayerBias,
            float[] nablaBias
            )
        {
            if (currentLayerWeights == null)
            {
                throw new ArgumentNullException("currentLayerWeights");
            }
            if (nablaWeights == null)
            {
                throw new ArgumentNullException("nablaWeights");
            }

            Parallel.For(0, currentLayerWeights.Length, cc =>
            //for (var cc = 0; cc < currentLayerWeights.Length; cc++)
            {
                currentLayerWeights[cc] += nablaWeights[cc] / batchSize;
            }
            ); //Parallel.For

            Parallel.For(0, currentLayerBias.Length, cc =>
            //for (var cc = 0; cc < currentLayerBias.Length; cc++)
            {
                currentLayerBias[cc] += nablaBias[cc] / batchSize;
            }
            ); //Parallel.For
        }
    }
}
