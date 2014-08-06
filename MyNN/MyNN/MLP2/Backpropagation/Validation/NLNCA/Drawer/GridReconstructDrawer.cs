using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.Data.Visualizer;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer
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

            if (_validationData.IsAuencoderDataSet)
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
                for (var cc = startIndex; cc < startIndex + _visualizeAsPairCount; cc++)
                {
                    var i = new Pair<float[], float[]>(
                        _validationData[cc].Input,
                        netResults[cc].NState);
                    pairList.Add(i);
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
