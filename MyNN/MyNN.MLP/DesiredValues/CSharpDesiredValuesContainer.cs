using System;
using System.Linq;
using MyNN.MLP.Structure;

namespace MyNN.MLP.DesiredValues
{
    public class CSharpDesiredValuesContainer : ICSharpDesiredValuesContainer
    {
        public float[] DesiredOutput
        {
            get;
            private set;
        }

        public CSharpDesiredValuesContainer(
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var outputLength = mlp.Layers.Last().NonBiasNeuronCount;

            this.DesiredOutput = new float[outputLength];
        }

        public void SetValues(float[] desiredValues)
        {
            if (desiredValues == null)
            {
                throw new ArgumentNullException("desiredValues");
            }
            if (desiredValues.Length != DesiredOutput.Length)
            {
                throw new InvalidOperationException("desiredValues.Length != DesiredOutput.Length");
            }

            desiredValues.CopyTo(DesiredOutput, 0);
        }

    }
}