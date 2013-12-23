using System;
using System.Collections;
using System.Collections.Generic;

namespace MyNN.MLP2.Structure
{
    public class LayerState : ILayerState
    {
        public float[] State
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

            State = new float[takeCount];
            Array.Copy(state, 0, State, 0, takeCount);
        }

        public IEnumerator<float> GetEnumerator()
        {
            var e = this.State.GetEnumerator();

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