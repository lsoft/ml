using System;
using System.Collections.Generic;
using MyNN;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.PingPong
{
    public class NextDataSet
    {
        public static void NextDataSets(
            string mlpPath,
            DataSet trainData,
            DataSet validationData,
            out DataSet trainNext,
            out DataSet validationNext)
        {
            var mlp = new SerializationHelper().LoadFromFile<MLP>(mlpPath);

            mlp.CutLastLayer();

            Console.WriteLine("Network configuration: " + mlp.GetLayerInformation());

            List<ILayerState> trainOutput;
            List<ILayerState> validationOutput;
            using (var clProvider = new CLProvider())
            {
                var forward = new CPUForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    mlp,
                    clProvider);

                trainOutput = forward.ComputeOutput(trainData);
                validationOutput = forward.ComputeOutput(validationData);
            }

            var trainItems = new List<DataItem>();
            for (var ti = 0; ti < trainOutput.Count; ti++)
            {
                var item =
                    new DataItem(
                        trainOutput[ti].State,
                        trainData[ti].Output);

                trainItems.Add(item);
            }

            trainNext = new DataSet(
                trainItems,
                trainData.Visualizer);

            var validationItems = new List<DataItem>();
            for (var ti = 0; ti < validationOutput.Count; ti++)
            {
                var item =
                    new DataItem(
                        validationOutput[ti].State,
                        validationData[ti].Output);

                validationItems.Add(item);
            }

            validationNext = new DataSet(
                validationItems,
                validationData.Visualizer);

        }
    }
}