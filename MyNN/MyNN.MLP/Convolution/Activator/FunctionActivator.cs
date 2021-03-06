﻿using System;
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
            if (currentNet.SpatialDimension.Multiplied != currentState.SpatialDimension.Multiplied)
            {
                throw new ArgumentException("currentNet.SpatialDimension.Multiplied != currentState.SpatialDimension.Multiplied");
            }
            if (currentNet.Width != currentState.Width)
            {
                throw new ArgumentException("currentNet.Width != currentState.Width");
            }
            if (currentNet.Height != currentState.Height)
            {
                throw new ArgumentException("currentNet.Height != currentState.Height");
            }

            //применяем функцию активации
            Parallel.For(0, currentNet.Height, j =>
            //for (var j = 0; j < currentNet.Height; j++)
            {
                for (var i = 0; i < currentNet.Width; i++)
                {
                    var net = currentNet.GetValueFromCoordSafely(i, j);

                    var state = activationFunction.Compute(net);

                    currentState.SetValueFromCoordSafely(i, j, state);
                }
            }
            ); //Parallel.For
        }
    }
}
