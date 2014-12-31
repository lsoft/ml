using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.NLNCA.Backpropagation.Validation.NLNCA
{
    public class NLNCADrawer : IDrawer
    {
        private readonly IArtifactContainer _bitmapContainer;
        private readonly IDataSet _validationData;
        private readonly int? _epochNumber;

        private readonly Color[] _colors;

        private readonly List<float[]> _grid;

        public NLNCADrawer(
            IDataSet validationData,
            IArtifactContainer artifactContainer,
            IColorProvider colorProvider,
            int? epochNumber
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
            _epochNumber = epochNumber;

            _bitmapContainer = artifactContainer.GetChildContainer("bitmaps");
            _colors = colorProvider.GetColors();

            _grid = new List<float[]>();

       }

        public void SetSize(int netResultCount)
        {
            //nothing to do
        }

        public void DrawItem(
            ILayerState netResult
            )
        {
            if (netResult == null)
            {
                throw new ArgumentNullException("netResult");
            }

            if (netResult.NState.Length != 2)
            {
                return;
            }

            //так как этот визуализаторо применяется только в 2D, то
            //1) обычно это бывает в целях отладки на малых объемаъ
            //2) даже если итемов будет больше 50 000 000, то массив _grid
            //займет около гигабайта, что допустимо
            //таким образом, просто все сохраним в ОЗУ, а потом покажем "за раз"

            _grid.Add(netResult.NState);
        }

        public void Save()
        {
            if (_grid.Count > 0)
            {
                //рисуем на картинке
                var maxx = _grid.Max(j => j[0]);
                var minx = _grid.Min(j => j[0]);
                var maxy = _grid.Max(j => j[1]);
                var miny = _grid.Min(j => j[1]);

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

                    foreach (var pair in _grid.ZipEqualLength(_validationData))
                    {
                        var netResult = pair.Value1;
                        var testItem = pair.Value2;

                        var ox = netResult[0];
                        var oy = netResult[1];

                        var x = (ox - minx)*(imageWidth - 1)/(maxx - minx);
                        var y = (oy - miny)*(imageHeight - 1)/(maxy - miny);

                        g.DrawRectangle(
                            new Pen(_colors[testItem.OutputIndex]),
                            (int) x, (int) y, 1, 1
                            );
                        g.DrawRectangle(
                            new Pen(_colors[testItem.OutputIndex]),
                            (int) x, (int) y, 2, 2
                            );
                        g.DrawRectangle(
                            new Pen(_colors[testItem.OutputIndex]),
                            (int) x, (int) y, 3, 3
                            );
                        ii++;
                    }
                }

                using (var s = _bitmapContainer.GetWriteStreamForResource(
                    string.Format(
                        "{0}.bmp",
                        _epochNumber != null ? _epochNumber.Value.ToString() : "_pretrain")))
                {
                    bitmap.Save(s, System.Drawing.Imaging.ImageFormat.Bmp);

                    s.Flush();
                }
            }
        }
    }
}
