using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.Activator
{
    public class FunctionActivator : IFunctionActivator
    {
        public void Apply(
            IFunction activationFunction,
            IReferencedSquareFloat currentNet,
            IReferencedSquareFloat currentState
            )
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }
            if (currentNet == null)
            {
                throw new ArgumentNullException("currentNet");
            }
            if (currentState == null)
            {
                throw new ArgumentNullException("currentState");
            }

            //применяем функцию активации
            for (var i = 0; i < currentNet.Width; i++)
            {
                for (var j = 0; j < currentNet.Height; j++)
                {
                    var net = currentNet.GetValueFromCoordSafely(i, j);

                    var state = activationFunction.Compute(net);

                    currentState.SetValueFromCoordSafely(i, j, state);
                }
            }
        }
    }
}
