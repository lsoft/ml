using System;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer
{
    public interface IEpocheTrainer
    {
        IForwardPropagation ForwardPropagation
        {
            get;
        }

        void PreTrainInit(DataSet data);

        void TrainEpoche(
            DataSet data,
            string epocheRoot,
            float learningRate);
    }
}