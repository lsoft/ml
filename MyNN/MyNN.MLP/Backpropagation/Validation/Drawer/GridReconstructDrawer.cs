using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Visualizer;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.Drawer
{
    public class GridReconstructDrawer : IDrawer
    {
        private readonly IVisualizer _visualizer;
        private readonly IDataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;

        public GridReconstructDrawer(
            IVisualizer visualizer,
            IDataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount
            )
        {
            if (visualizer == null)
            {
                throw new ArgumentNullException("visualizer");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _visualizer = visualizer;
            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
        }

        public void Draw(
            IArtifactContainer containerForSave,
            int? epocheNumber, 
            List<ILayerState> netResults)
        {
            if (containerForSave == null)
            {
                throw new ArgumentNullException("containerForSave");
            }
            if (netResults == null)
            {
                throw new ArgumentNullException("netResults");
            }

            if (_validationData.IsAutoencoderDataSet)
            {
                using (var s = containerForSave.GetWriteStreamForResource("grid.bmp"))
                {
                    _visualizer.SaveAsGrid(
                        s,
                        netResults.ConvertAll(j => j.NState).Take(_visualizeAsGridCount).ToList());

                    s.Flush();
                }

                //со случайного индекса
                var startIndex = (int) ((DateTime.Now.Millisecond/1000f)*(Math.Min(_validationData.Count, netResults.Count) - _visualizeAsPairCount));

                var pairList = new List<Pair<float[], float[]>>();
                //for (var cc = startIndex; cc < startIndex + _visualizeAsPairCount; cc++)
                //{
                //    var i = new Pair<float[], float[]>(
                //        _validationData.Data[cc].Input,
                //        netResults[cc].NState);
                //    pairList.Add(i);
                //}
                foreach (var pair in netResults.ZipEqualLength(_validationData).Skip(startIndex).Take(_visualizeAsPairCount))
                {
                    var netResult = pair.Value1;
                    var testItem = pair.Value2;

                    pairList.Add(
                        new Pair<float[], float[]>(
                            testItem.Output,
                            netResult.NState));
                }

                using (var s = containerForSave.GetWriteStreamForResource("reconstruct.bmp"))
                {
                    _visualizer.SaveAsPairList(
                        s,
                        pairList);

                    s.Flush();
                }
            }
        }
    }
}
