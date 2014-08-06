using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer
{
    public class NLNCADrawer : IDrawer
    {
        private readonly IArtifactContainer _bitmapContainer;
        private readonly IDataSet _validationData;

        private readonly Color[] _colors;

        public NLNCADrawer(
            IDataSet validationData,
            IArtifactContainer artifactContainer,
            IColorProvider colorProvider
            )
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }

            _validationData = validationData;

            _bitmapContainer = artifactContainer.GetChildContainer("bitmaps");
            _colors = colorProvider.GetColors();
        }

        public void Draw(
            IArtifactContainer containerForSave, 
            int? epocheNumber,
            List<ILayerState> netResults
            )
        {
            if (containerForSave == null)
            {
                throw new ArgumentNullException("containerForSave");
            }
            if (netResults == null)
            {
                throw new ArgumentNullException("netResults");
            }


            if (netResults[0].NState.Length != 2)
            {
                return;
            }

            //рисуем на картинке
            var maxx = netResults.Max(j => j.NState[0]);
            var minx = netResults.Min(j => j.NState[0]);
            var maxy = netResults.Max(j => j.NState[1]);
            var miny = netResults.Min(j => j.NState[1]);

            const int imageWidth = 500;
            const int imageHeight = 500;

            var bitmap = new Bitmap(imageWidth, imageHeight);
            var ii = 0;

            using (var g = Graphics.FromImage(bitmap))
            {
                g.Clear(Color.White);

                g.DrawString(
                    minx.ToString() + ";" + miny.ToString(),
                    new Font("Tahoma", 12),
                    Brushes.Black,
                    0, 0);

                g.DrawString(
                    maxx.ToString() + ";" + maxy.ToString(),
                    new Font("Tahoma", 12),
                    Brushes.Black,
                    300, 450);

                foreach (var netResult in netResults)
                {
                    var ox = netResult.NState[0];
                    var oy = netResult.NState[1];

                    var x = (ox - minx)*(imageWidth - 1)/(maxx - minx);
                    var y = (oy - miny)*(imageHeight - 1)/(maxy - miny);

                    g.DrawRectangle(
                        new Pen(_colors[_validationData[ii].OutputIndex]),
                        (int) x, (int) y, 1, 1
                        );
                    g.DrawRectangle(
                        new Pen(_colors[_validationData[ii].OutputIndex]),
                        (int) x, (int) y, 2, 2
                        );
                    g.DrawRectangle(
                        new Pen(_colors[_validationData[ii].OutputIndex]),
                        (int) x, (int) y, 3, 3
                        );
                    ii++;
                }
            }

            using (var s = _bitmapContainer.GetWriteStreamForResource(
                string.Format(
                    "{0}.bmp",
                    epocheNumber != null ? epocheNumber.Value.ToString() : "_pretrain")))
            {
                bitmap.Save(s, System.Drawing.Imaging.ImageFormat.Bmp);

                s.Flush();
            }
        }
    }
}
