using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Randomizer;
using OpenCvSharp;

namespace MyNN.Common.Data.DataLoader
{
    /// <summary>
    /// Класс, размножающий множество MNIST картинок с помощью афинных и эластичных трансформаций
    /// </summary>
    public class MNISTElasticExpander
    {
        private readonly IRandomizer _randomizer;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IDataSetFactory _dataSetFactory;

        public MNISTElasticExpander(
            IRandomizer randomizer,
            IDataItemFactory dataItemFactory,
            IDataSetFactory dataSetFactory
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (dataSetFactory == null)
            {
                throw new ArgumentNullException("dataSetFactory");
            }

            _randomizer = randomizer;
            _dataItemFactory = dataItemFactory;
            _dataSetFactory = dataSetFactory;
        }

        public IDataSet GenerateExpandedSet(
            IDataSet toDeformDataSet,
            int deformationEpocheCount,
            int stepRefreshGaussMap
            )
        {

            Console.WriteLine("Deformations starts...");

            var originalImagesCount = toDeformDataSet.Count;
            var newImagesCount = originalImagesCount * (deformationEpocheCount + 1);

            var resultItemList = new List<IDataItem>(newImagesCount + 100);

            for (var dd = 0; dd < deformationEpocheCount; dd++)
            {
                var start = DateTime.Now;
                Console.WriteLine("Deformations epoche: " + dd);

                //Запускаем операцию искажения, последовательно, так как каждые stepRefreshGaussMap итераций
                //мы будет менять гауссово поле искажений
                var cc = 0;
                foreach (var o in toDeformDataSet.Take(originalImagesCount))
                {
                    //какое это число?
                    var correctValue = o.Output.ToList().FindIndex(j => j > 0.0);
                    var maxAngle = (correctValue == 1 || correctValue == 7) ? 6 : 15;

                    bool changed;
                    var d0 = LetterAffineDeform(
                        _randomizer,
                        o.Input,
                        maxAngle,
                        28,
                        28,
                        out changed);

                    if (cc % stepRefreshGaussMap == 0)
                    {
                        //каждую stepRefreshGaussMap операцию генерируем новое гауссово поле искажений
                        CreateGaussedRandomDisplacementMap(
                            _randomizer,
                            28,
                            28);
                    }

                    var d1 = LetterElasticDeform2(
                        d0,
                        28,
                        28,
                        2);

                    //if (changed)
                    {
                        resultItemList.Add(
                            _dataItemFactory.CreateDataItem(
                                d1, 
                                o.Output));
                    }

                    cc++;
                }

                var end = DateTime.Now;
                Console.WriteLine("Epoche takes " + (end - start));
            }

            Console.WriteLine("Deformations finished...");

            return
                _dataSetFactory.CreateDataSet(
                    new FromArrayDataItemLoader(
                        resultItemList,
                        new DefaultNormalizer()),
                    0
                    );
        }


        private readonly object _gaussLocker = new object();
        private volatile double[] _gaussCoef;
        private double[,] _gaussX, _gaussY;
        private double _maxXGauss, _minXGauss, _maxYGauss, _minYGauss;

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

            #region показываем карту искажений

            /*
            //show the deformation maps

            var rdfXImage = new IplImage(new CvSize(scaledImageWidth, scaledImageHeight), BitDepth.U8, 1);
            var gaussXImage = new IplImage(new CvSize(scaledImageWidth, scaledImageHeight), BitDepth.U8, 1);

            for (var x = 0; x < scaledImageWidth; x++)
            {
                for (var y = 0; y < scaledImageHeight; y++)
                {
                    var value = Math.Abs(_gaussX[x, y]) * (255.0 / _maxXGauss / deformXScale);

                    gaussXImage.Set2D(
                        x,
                        y,
                        new CvScalar(
                            (int)value,
                            (int)value,
                            (int)value));

                    var value2 = Math.Abs(rdfX[x, y]) * 255;
                    rdfXImage.Set2D(
                        x,
                        y,
                        new CvScalar(
                            (int)value2,
                            (int)value2,
                            (int)value2));
                }
            }


            using (var w1 = new CvWindow("random map"))
            {
                w1.ShowImage(rdfXImage);

                using (var w2 = new CvWindow("gauss random map"))
                {
                    w2.ShowImage(gaussXImage);
                    Cv.WaitKey();
                }
            }
            //*/

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

                        //using (var w2 = new CvWindow("b"))
                        //{
                        //    w2.ShowImage(gray);
                        //    Cv.WaitKey();
                        //}
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

                        #region вычисление и сдвиг по центру масс (убрано, так как вроде хуже получается)

                        //int massX, massY;
                        //FindCenterOfMass(ds, out massX, out massY);

                        ////определим центр картинки
                        //var imageCenterX = imageWidth / 2;
                        //var imageCenterY = imageHeight / 2;

                        ////определим смещения от центра картинки
                        //var diffX = imageCenterX - massX;
                        //var diffY = imageCenterY - massY;

                        //if (diffX > 0 || diffY > 0)
                        //{
                        //    //смещать надо
                        //    Console.WriteLine("diffX=" + diffX + ",   diffY=" + diffY);

                        //    using (var shifted = new IplImage(ds.Size, BitDepth.U8, 1))
                        //    {
                        //        Cv.Zero(shifted);
                        //        //Cv.Set(shifted, 255);

                        //        var origLeft = diffX > 0 ? 0 : diffX;
                        //        var origTop = diffY > 0 ? 0 : diffY;
                        //        var width = imageWidth - diffX;
                        //        var height = imageHeight - diffY;

                        //        if (origLeft + width > imageWidth)
                        //        {
                        //            width -= (imageWidth - (origLeft + width));
                        //        }

                        //        if (origTop + height > imageHeight)
                        //        {
                        //            height -= (imageHeight - (origTop + height));
                        //        }

                        //        var targetLeft = diffX > 0 ? diffX : 0;
                        //        var targetTop = diffY > 0 ? diffY : 0;

                        //        ds.SetROI(new CvRect(origLeft, origTop, width, height));
                        //        shifted.SetROI(new CvRect(targetLeft, targetTop, width, height));
                        //        Cv.Copy(ds, shifted);
                        //        shifted.ResetROI();
                        //        ds.ResetROI();

                        //        //using (var w2 = new CvWindow("ds"))
                        //        //{
                        //        //    w2.ShowImage(ds);

                        //        //    using (var w3 = new CvWindow("shifted"))
                        //        //    {
                        //        //        w3.ShowImage(shifted);
                        //        //        Cv.WaitKey();
                        //        //    }
                        //        //}

                        //        //недоделано: сдвиг то считается, но для формирования результата
                        //        //все равно используется ds (см. ниже заполнение массива downResult)
                        //    }
                        //}

                        #endregion

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

                        //using (var w2 = new CvWindow("gray"))
                        //{
                        //    w2.ShowImage(gray);
                        //    Cv.WaitKey();
                        //}

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
                //using (var w2 = new CvWindow("44"))
                //{
                //    w2.ShowImage(src);
                //    Cv.WaitKey();
                //}

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

                            //if (rnd.NextDouble() > 0.5)
                            //{
                            //    var orig_left = 0;
                            //    var orig_top = 0;
                            //    var orig_width = src.Width;
                            //    var orig_height = src.Height;

                            //    var target_left = 0;
                            //    var target_top = 0;
                            //    var target_width = src.Width;
                            //    var target_height = src.Height;

                            //    var hShift = (int)(rnd.NextDouble() * 3);
                            //    orig_width -= hShift;
                            //    target_width -= hShift;

                            //    var vShift = (int)(rnd.NextDouble() * 3);
                            //    orig_height -= vShift;
                            //    target_height -= vShift;

                            //    if (rnd.NextDouble() > 0.5)
                            //    {
                            //        //сдвиг вправо
                            //        target_left += hShift;
                            //    }
                            //    else
                            //    {
                            //        //сдвиг влево
                            //        orig_left += hShift;
                            //    }

                            //    if (rnd.NextDouble() > 0.5)
                            //    {
                            //        //сдвиг вниз
                            //        target_top += vShift;
                            //    }
                            //    else
                            //    {
                            //        //сдвиг вверх
                            //        orig_top += vShift;
                            //    }

                            //    rotated.SetROI(new CvRect(orig_left, orig_top, orig_width, orig_height));
                            //    shifted.SetROI(new CvRect(target_left, target_top, target_width, target_height));
                            //    Cv.Copy(rotated, shifted);
                            //    shifted.ResetROI();
                            //    rotated.ResetROI();

                            //    changed = true;
                            //}
                            //else
                            {
                                Cv.Copy(rotated, shifted);
                            }

                            //using (var w2 = new CvWindow("44"))
                            //{
                            //    w2.ShowImage(shifted);
                            //    Cv.WaitKey();
                            //}

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
    }
}
