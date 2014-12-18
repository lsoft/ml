using System;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.Noiser
{
    /// <summary>
    /// Эластичный нойзер
    /// </summary>
    [Serializable]
    public class ElasticNoiser : INoiser
    {
        private readonly IRandomizer _randomizer;
        private readonly int _stepRefreshGaussMap;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly bool _isNeedToRenormalizeTo01;

        private readonly object _gaussLocker = new object();
        private volatile double[] _gaussCoef;
        private double[,] _gaussX, _gaussY;
        private double _maxXGauss, _minXGauss, _maxYGauss, _minYGauss;

        private int _noisedCount;

        public ElasticNoiser(
            IRandomizer randomizer,
            int stepRefreshGaussMap,
            int imageWidth,
            int imageHeight,
            bool isNeedToRenormalizeTo01
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
            _stepRefreshGaussMap = stepRefreshGaussMap;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
            _isNeedToRenormalizeTo01 = isNeedToRenormalizeTo01;

            _noisedCount = 0;
        }

        public float[] ApplyNoise(
            float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            const int maxAngle = 6;

            bool changed;
            var d0 = LetterAffineDeform(
                _randomizer,
                data,
                maxAngle,
                _imageWidth,
                _imageHeight,
                out changed);

            if (_noisedCount++ % _stepRefreshGaussMap == 0)
            {
                //каждую stepRefreshGaussMap операцию генерируем новое гауссово поле искажений
                CreateGaussedRandomDisplacementMap(
                    _randomizer,
                    _imageWidth,
                    _imageHeight
                    );
            }

            var d1 = LetterElasticDeform2(
                d0,
                _imageWidth,
                _imageHeight,
                2);

            if (_isNeedToRenormalizeTo01)
            {
                var min = d1.Min();
                d1.Transform((a) => a - min);

                var max = d1.Max();
                var coef = 1f/max;
                d1.Transform((a) => a * coef);
            }

            return d1;
        }

        private void CreateGaussedRandomDisplacementMap(
            IRandomizer randomizer,
            int imageWidth,
            int imageHeight,
            int scaleFactor = 2,
            double maxDeformScale = 4.0, //max deformation scale in pixels
            int radius = 32, //gauss radius
            double sigma = 10.0 //gauss sigma
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            #region генерируем массивы гаусса

            //генерируем массивы гаусса
            if (_gaussCoef == null)
            {
                lock (_gaussLocker)
                {
                    if (_gaussCoef == null)
                    {
                        _gaussCoef = new double[2 * (radius + 1) * (radius + 1)];

                        for (var dx = 0; dx < radius; dx++)
                        {
                            for (var dy = 0; dy < radius; dy++)
                            {
                                var index = dx * dx + dy * dy;

                                _gaussCoef[index] =
                                    (1.0 / (2.0 * Math.PI * sigma * sigma))
                                    * Math.Exp(-index / (2.0 * sigma * sigma));
                            }
                        }
                    }
                }
            }

            #endregion

            var scaledImageWidth = imageWidth * scaleFactor;
            var scaledImageHeight = imageHeight * scaleFactor;

            #region генерируем шумовую карту

            //random displacement field

            var rdfX = new double[scaledImageWidth, scaledImageHeight];
            var rdfY = new double[scaledImageWidth, scaledImageHeight];

            const int step = 4;

            for (var x = 0; x < scaledImageWidth; x += step)
            {
                for (var y = 0; y < scaledImageHeight; y += step)
                {
                    var diffx = randomizer.Next() * 2.0 - 1.0;
                    rdfX[x, y] = diffx;

                    var diffy = randomizer.Next() * 2.0 - 1.0;
                    rdfY[x, y] = diffy;
                }
            }

            #endregion

            #region смазываем шумовую карту

            //gaussed random displacement field

            _gaussX = new double[scaledImageWidth, scaledImageHeight];
            _gaussY = new double[scaledImageWidth, scaledImageHeight];

            Parallel.For(0, scaledImageWidth, x =>
            //for (var x = 0; x < scaledImageWidth; x++)
            {
                for (var y = 0; y < scaledImageHeight; y++)
                {
                    var deltaGaussX = 0.0;
                    var deltaGaussY = 0.0;

                    for (var dx = -radius + 1; dx < radius; dx++)
                    {
                        for (var dy = -radius + 1; dy < radius; dy++)
                        {
                            if (Math.Sqrt(dx * dx + dy * dy) >= radius)
                                continue;

                            var xc = x + dx;
                            var yc = y + dy;

                            var index = dx * dx + dy * dy;
                            var gaussCoef = _gaussCoef[index];

                            var rdf_x =
                                xc < 0 || xc >= scaledImageWidth
                                || yc < 0 || yc >= scaledImageHeight
                                    ? 0.0
                                    : rdfX[xc, yc];

                            var rdf_y =
                                xc < 0 || xc >= scaledImageWidth
                                || yc < 0 || yc >= scaledImageHeight
                                    ? 0.0
                                    : rdfY[xc, yc];

                            deltaGaussX += rdf_x * gaussCoef;
                            deltaGaussY += rdf_y * gaussCoef;
                        }
                    }

                    _gaussX[x, y] = deltaGaussX;
                    _gaussY[x, y] = deltaGaussY;

                }
            }
            ); //Parallel.For

            #endregion

            #region ищем максимум-минимум в карте

            //max, min collect
            _maxXGauss = double.MinValue;
            _minXGauss = double.MaxValue;
            _maxYGauss = double.MinValue;
            _minYGauss = double.MaxValue;
            for (var x = 0; x < scaledImageWidth; x++)
            {
                for (var y = 0; y < scaledImageHeight; y++)
                {
                    var xValue = Math.Abs(_gaussX[x, y]);
                    if (_maxXGauss < xValue)
                    {
                        _maxXGauss = xValue;
                    }
                    if (_minXGauss > xValue && xValue > 0)
                    {
                        _minXGauss = xValue;
                    }

                    var yValue = Math.Abs(_gaussY[x, y]);
                    if (_maxYGauss < yValue)
                    {
                        _maxYGauss = yValue;
                    }
                    if (_minYGauss > yValue && yValue > 0)
                    {
                        _minYGauss = yValue;
                    }
                }
            }

            #endregion

            #region выравниваем масштаб карты искажений

            //apply defined scale of deformation
            var deformXScale = maxDeformScale / _maxXGauss;
            var deformYScale = maxDeformScale / _maxYGauss;
            for (var x = 0; x < scaledImageWidth; x++)
            {
                for (var y = 0; y < scaledImageHeight; y++)
                {
                    _gaussX[x, y] *= deformXScale;
                    _gaussY[x, y] *= deformYScale;
                }
            }

            #endregion
        }

        private unsafe float[] LetterElasticDeform2(
            float[] imageBuffer,
            int imageWidth,
            int imageHeight,
            int scaleFactor)
        {
            var origByte = Array.ConvertAll(imageBuffer, j => (byte)(j * 255f));

            var scaledImageWidth = imageWidth * scaleFactor;
            var scaledImageHeight = imageHeight * scaleFactor;

            var scaledImageBuffer = new byte[scaledImageWidth * scaledImageHeight];

            fixed (byte* p = origByte)
            {
                using (var gray = IplImage.FromPixelData(imageWidth, imageHeight, 1, new IntPtr(p)))
                {
                    using (var rs = new IplImage(scaledImageWidth, scaledImageHeight, BitDepth.U8, 1))
                    {
                        gray.Resize(rs, Interpolation.Lanczos4);

                        //записываем обратно в буфер
                        for (var x = 0; x < scaledImageWidth; x += 1)
                        {
                            for (var y = 0; y < scaledImageHeight; y += 1)
                            {
                                scaledImageBuffer[scaledImageHeight * x + y] = (byte)(rs.Get2D(x, y).Val0);
                            }
                        }
                    }
                }
            }

            //обработка буквы

            var result = new byte[scaledImageWidth * scaledImageHeight];

            //apply gaussed random displacement field

            for (var x = 0; x < scaledImageWidth; x++)
            {
                for (var y = 0; y < scaledImageHeight; y++)
                {
                    var shiftX = _gaussX[x, y];
                    var shiftY = _gaussY[x, y];

                    var newX = (int)(x + shiftX + 0.5f);
                    var newY = (int)(y + shiftY + 0.5f);

                    var newPixel =
                        newX < 0 || newX >= scaledImageWidth
                        || newY < 0 || newY >= scaledImageHeight
                            ? (byte)0
                            : scaledImageBuffer[scaledImageWidth * newY + newX];

                    result[scaledImageHeight * x + y] = newPixel;
                }
            }

            //downsample scaled

            fixed (byte* p = result)
            {
                using (var gray = IplImage.FromPixelData(scaledImageWidth, scaledImageHeight, 1, new IntPtr(p)))
                {
                    using (var ds = new IplImage(imageWidth, imageHeight, BitDepth.U8, 1))
                    {
                        Cv.Resize(gray, ds, Interpolation.Lanczos4);

                        var downResult = new float[imageWidth * imageHeight];

                        var scaledIndex = 0;
                        for (var y = 0; y < imageHeight; y += 1)
                        {
                            for (var x = 0; x < imageWidth; x += 1)
                            {
                                downResult[scaledIndex] = (float)(ds.Get2D(x, y).Val0 / 255.0f);

                                scaledIndex++;
                            }
                        }

                        return
                            downResult;
                    }
                }
            }
        }


        private float[] LetterAffineDeform(
            IRandomizer randomizer,
            float[] imageBuffer,
            int maxRotationAngle,
            int imageWidth,
            int imageHeight,
            out bool changed)
        {
            var origImage = new Bitmap(imageWidth, imageHeight);

            var inImageIndex = 0;
            for (var h = 0; h < imageHeight; h++)
            {
                for (var w = 0; w < imageWidth; w++)
                {
                    var value = imageBuffer[inImageIndex];
                    origImage.SetPixel(
                        w,
                        h,
                        Color.FromArgb(
                            (int) (value*255),
                            (int) (value*255),
                            (int) (value*255)));

                    inImageIndex++;
                }
            }

            //обработка буквы
            using (var src = BitmapConverter.ToIplImage(origImage))
            {
                using (var gray = new IplImage(src.Size, BitDepth.U8, 1))
                {
                    Cv.CvtColor(src, gray, ColorConversion.RgbToGray);

                    changed = false;

                    if (randomizer.Next() > 0.8f)
                    {
                        gray.Dilate(gray);
                        gray.Erode(gray);

                        changed = true;
                    }

                    using (var rotated = new IplImage(src.Size, BitDepth.U8, 1))
                    {
                        if (randomizer.Next() > 0.2f)
                        {
                            var centerShiftX = randomizer.Next() * 20 - 10;
                            var centerShiftY = randomizer.Next() * 20 - 10;
                            var angle = randomizer.Next() * (2 * maxRotationAngle) - maxRotationAngle;

                            var rotateCenter = new CvPoint2D32f(src.Width/2f + centerShiftX, src.Height/2f + centerShiftY);
                            var rotateMatrix = Cv.GetRotationMatrix2D(rotateCenter, angle, 1.0);

                            rotated.Set(new CvScalar(0));
                            Cv.WarpAffine(gray, rotated, rotateMatrix);

                            changed = true;
                        }
                        else
                        {
                            Cv.Copy(gray, rotated);
                        }

                        using (var shifted = new IplImage(src.Size, BitDepth.U8, 1))
                        {
                            shifted.Zero();

                            Cv.Copy(rotated, shifted);

                            var result = new float[imageWidth*imageHeight];

                            var inResultImageIndex = 0;
                            for (var w = 0; w < imageWidth; w++)
                            {
                                for (var h = 0; h < imageHeight; h++)
                                {
                                    var value3 =
                                        shifted.Get2D(w, h);
                                    var value = Math.Sqrt((value3.Val0*value3.Val0 + value3.Val1*value3.Val1 + value3.Val2*value3.Val2)/3.0);

                                    result[inResultImageIndex] = (float) (value/255.0f);

                                    inResultImageIndex++;
                                }
                            }

                            return result;
                        }
                    }
                }
            }
        }

        #region viewer class

        public class ElasticNoiserMNISTViewer
        {
            public void DoView(
                IDataSet dataset)
            {
                if (dataset == null)
                {
                    throw new ArgumentNullException("dataset");
                }

                var noiser = new ElasticNoiser(
                    new DefaultRandomizer(1234),
                    1,
                    28,
                    28,
                    true
                    );

                foreach (var d in dataset)
                {
                    var origFloat = noiser.ApplyNoise(d.Input);
                    var origByte = origFloat.ConvertAll<float, byte>((a) => (byte)(a * 255));

                    unsafe
                    {
                        fixed (byte* p = origByte)
                        {
                            using (var gray = IplImage.FromPixelData(28, 28, 1, new IntPtr(p)))
                            {
                                using (var w2 = new CvWindow("b"))
                                {
                                    w2.ShowImage(gray);
                                    Cv.WaitKey();
                                }
                            }
                        }
                    }
                }
            }
        }

        #endregion

    }
}
