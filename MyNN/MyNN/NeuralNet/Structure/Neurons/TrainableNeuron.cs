using System;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure.Neurons
{
    [Serializable]
    public abstract class TrainableNeuron
        //: ITrainableNeuron
    {
        public float Dedz
        {
            get;
            set;
        }

        public float[] Weights
        {
            get;
            protected set;
        }

        public float LastState
        {
            get;
            set;
        }

        public float LastNET
        {
            get;
            protected set;
        }

        public IFunction ActivationFunction
        {
            get;
            protected set;
        }

        /// <summary>
        /// Метод используется для модификации состояния нейрона при прямом просчете
        /// сети нестандартным алгоритмом
        /// </summary>
        public void SetState(float lastNET, float lastState, float dedz)
        {
            this.LastNET = lastNET;
            this.LastState = lastState;
            this.Dedz = dedz;
        }


        public abstract float Activate(float[] inputVector);
    }
}
