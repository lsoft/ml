using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine
{
    [Serializable]
    public class SaveableContainer
    {
        public float[] Weights
        {
            get;
            private set;
        }

        public float[] VisibleBiases
        {
            get;
            private set;
        }

        public float[] HiddenBiases
        {
            get;
            private set;
        }

        public SaveableContainer(
            float[] weights, 
            float[] visibleBiases, 
            float[] hiddenBiases
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visibleBiases == null)
            {
                throw new ArgumentNullException("visibleBiases");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }

            Weights = weights;
            VisibleBiases = visibleBiases;
            HiddenBiases = hiddenBiases;
        }
    }
}
