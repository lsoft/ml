using System;
using System.Collections;
using System.Collections.Generic;

namespace MyNN.MLP.Structure.Layer
{
    public class LayerState : ILayerState
    {
        public float[] NState
        {
            get;
            private set;
        }

        public LayerState(float[] state, int takeCount)
        {
            if (state == null)
            {
                throw new ArgumentNullException("state");
            }

            NState = new float[takeCount];
            Array.Copy(state, 0, NState, 0, takeCount);
        }

        public IEnumerator<float> GetEnumerator()
        {
            var e = this.NState.GetEnumerator();

            while (e.MoveNext())
                yield return (float)e.Current;

            //return (IEnumerator<float>) e;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}