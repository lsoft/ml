using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Visualizer.Factory;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.ForwardPropagationFactory;
using MyNN.MLP.Structure;

namespace MyNN.MLP
{
    public class FileSystemFeatureVisualization
    {
        private readonly IMLP _mlp;
        private readonly IForwardPropagationFactory _forwardPropagationFactory;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IRandomizer _randomizer;

        public FileSystemFeatureVisualization(
            IRandomizer randomizer,
            IMLP mlp,
            IForwardPropagationFactory forwardPropagationFactory,
            IDataItemFactory dataItemFactory
            )
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
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            _randomizer = randomizer;
            _mlp = mlp;
            _forwardPropagationFactory = forwardPropagationFactory;
            _dataItemFactory = dataItemFactory;
        }

        public void Visualize(
            IVisualizerFactory visualizerFactory,
            string imageFileName,
            int nonZeroCount,
            float featureValue,
            int takeIntoAccount = 900,
            bool randomOrder = true,
            bool clampTo01 = false
            )
        {
            if (visualizerFactory == null)
            {
                throw new ArgumentNullException("visualizerFactory");
            }
            if (imageFileName == null)
            {
                throw new ArgumentNullException("imageFileName");
            }

            if (File.Exists(imageFileName))
            {
                File.Delete(imageFileName);
            }

            _mlp.AutoencoderCutHead();

            ConsoleAmbientContext.Console.WriteLine(_mlp.GetLayerInformation());
            
            var forwardPropagation = _forwardPropagationFactory.Create(
                _randomizer,
                _mlp);

            var inputLayerSize = _mlp.Layers.First().NonBiasNeuronCount;

            var diList = new List<IDataItem>();
            for (var cc = 0; cc < inputLayerSize; cc++)
            {
                var input = new float[inputLayerSize];

                for (var nz = 0; nz < nonZeroCount; nz++)
                {
                    input[_randomizer.Next(input.Length)] = featureValue;
                }

                var output = new float[1];

                var di = _dataItemFactory.CreateDataItem(input, output);
                diList.Add(di);
            }

            var results = forwardPropagation.ComputeOutput(diList);

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
            var images = results.Take(takeIntoAccount).ToList().ConvertAll(j => j.NState);

            //if clamp is requested to do so
            if (clampTo01)
            {
                images.ForEach(j => j.Transform((f) => (f < 0f ? 0f : (f > 1f ? 1f : f))));
            }

            var visualizer = visualizerFactory.CreateVisualizer(
                images.Count
                );

            //visualize
            using (var s = new FileStream(imageFileName, FileMode.CreateNew, FileAccess.ReadWrite))
            {
                foreach (var image in images)
                {
                    visualizer.VisualizeGrid(
                        image
                        );
                }

                visualizer.SaveGrid(s);

                s.Flush();
            }
        }
    }
}
