using System;
using System.Collections.Generic;
using MyNN.Data;
using MyNN.NeuralNet.Train;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM
{
    public interface IRestrictedBoltzmannMachine
    {
        int RandomCount
        {
            get;
        }

        int VisibleNeuronCount
        {
            get;
        }

        int HiddenNeuronCount
        {
            get;
        }

        Random Random
        {
            get;
        }

        
        CLProvider CLProvider
        {
            get;
        }

        Mem<float> Weights
        {
            get;
        }

        Mem<float> Visible
        {
            get;
        }

        Mem<float> Hidden0
        {
            get;
        }

        Mem<float> Hidden1
        {
            get;
        }

        Mem<float> Randoms
        {
            get;
        }
        
        Kernel ComputeVisible
        {
            get;
        }

        Kernel SampleVisible
        {
            get;
        }
        
        Kernel ComputeHidden
        {
            get;
        }

        Kernel SampleHidden
        {
            get;
        }

        void CalculateFreeEnergy(
            string artifactFolderRoot,
            DataSet trainFreeEnergySet,
            DataSet validationFreeEnergySet);

        float CalculateFreeEnergySet(
            Mem<float> weights,
            DataSet data);

    }
}