using System;
using System.Collections.Generic;
using MyNN.Data;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

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

        IRandomizer Randomizer
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
            IDataSet trainFreeEnergySet,
            IDataSet validationFreeEnergySet);

        float CalculateFreeEnergySet(
            Mem<float> weights,
            IDataSet data);

    }
}