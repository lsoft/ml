using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.Data.Visualizer;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.ForwardPropagation.ForwardFactory;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2
{
    public class FeatureVisualization
    {
        private readonly MLP _mlp;
        private readonly IForwardPropagationFactory _forwardPropagationFactory;
        private readonly int _nonZeroCount;
        private readonly float _featureValue;
        private readonly IRandomizer _randomizer;

        public FeatureVisualization(
            IRandomizer randomizer,
            MLP mlp,
            IForwardPropagationFactory forwardPropagationFactory,
            int nonZeroCount,
            float featureValue)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (forwardPropagationFactory == null)
            {
                throw new ArgumentNullException("forwardPropagationFactory");
            }

            _randomizer = randomizer;
            _mlp = mlp;
            _forwardPropagationFactory = forwardPropagationFactory;
            _nonZeroCount = nonZeroCount;
            _featureValue = featureValue;
        }

        public void Visualize(
            IVisualizer visualizer,
            string imagefilename,
            int takeIntoAccount = 900,
            bool randomOrder = true,
            bool clampTo01 = false)
        {
            if (visualizer == null)
            {
                throw new ArgumentNullException("visualizer");
            }
            if (imagefilename == null)
            {
                throw new ArgumentNullException("imagefilename");
            }

            if (File.Exists(imagefilename))
            {
                File.Delete(imagefilename);
            }

            _mlp.AutoencoderCutHead();

            ConsoleAmbientContext.Console.WriteLine(_mlp.DumpLayerInformation());


            using (var clProvider = new CLProvider())
            {
                var forwardPropagation = _forwardPropagationFactory.Create(
                    _randomizer,
                    clProvider,
                    _mlp);

                var inputLayerSize = _mlp.Layers.First().NonBiasNeuronCount;

                var diList = new List<DataItem>();
                for (var cc = 0; cc < inputLayerSize; cc++)
                {
                    var input = new float[inputLayerSize];

                    for (var nz = 0; nz < _nonZeroCount; nz++)
                    {
                        input[_randomizer.Next(input.Length)] = _featureValue;
                    }

                    var output = new float[1];

                    var di = new DataItem(input, output);
                    diList.Add(di);
                }

                var artificalDataSet = new DataSet(diList);

                var results = forwardPropagation.ComputeOutput(artificalDataSet);

                if (randomOrder)
                {
                    //shuffle result if it requested
                    for (var cc = 0; cc < results.Count; cc++)
                    {
                        var index1 = _randomizer.Next(results.Count);
                        var index2 = _randomizer.Next(results.Count);

                        var u = results[index1];
                        results[index1] = results[index2];
                        results[index2] = u;
                    }
                }

                //cut to only requested count
                var images = results.Take(takeIntoAccount).ToList().ConvertAll(j => j.State);

                //if clamp is requested to do so
                if (clampTo01)
                {   
                    images.ForEach(j => j.Transform((f) => (f < 0f ? 0f : (f > 1f ? 1f : f))));
                }

                //visualize
                visualizer.SaveAsGrid(
                    imagefilename,
                    images);
            }
        }
    }
}
