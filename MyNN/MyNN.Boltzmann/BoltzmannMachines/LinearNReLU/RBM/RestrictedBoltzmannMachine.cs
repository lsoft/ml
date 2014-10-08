using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MyNN.Data;
using MyNN.Randomizer;


namespace MyNN.BoltzmannMachines.LinearNReLU.RBM
{
    public class RestrictedBoltzmannMachine
    {
        private readonly Normal _gaussRandom = new Normal(0, 1);

        public int VisibleNeuronCount
        {
            get;
            private set;
        }

        public int HiddenNeuronCount
        {
            get;
            private set;
        }

        private readonly IRandomizer _randomizer;
        private readonly int _imageWidth;
        private readonly int _imageHeight;

        public float[] VisibleNeuronBias
        {
            get;
            private set;
        }

        public float[] HiddenNeuronBias
        {
            get;
            private set;
        }

        public float[][] Weights //[hindex][vindex]
        {
            get;
            private set;
        }

        public string Name
        {
            get;
            private set;
        }

        public RestrictedBoltzmannMachine(
            IRandomizer randomizer,
            int visibleNeuronCount,
            int hiddenNeuronCount,
            int imageWidth,
            int imageHeight)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            VisibleNeuronCount = visibleNeuronCount;
            HiddenNeuronCount = hiddenNeuronCount;
            _randomizer = randomizer;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;

            //создаем массивы
            this.VisibleNeuronBias = new float[VisibleNeuronCount];
            this.HiddenNeuronBias = new float[HiddenNeuronCount];
            this.Weights = new float[HiddenNeuronCount][];
            for (var cc = 0; cc < HiddenNeuronCount; cc++)
            {
                this.Weights[cc] = new float[VisibleNeuronCount];
            }

            var grnd = new Normal(0.0, 0.01);
            //var rnd = new Random(123);

            //рандомизируем все
            for (var vi = 0; vi < VisibleNeuronCount; vi++)
            {
                this.VisibleNeuronBias[vi] =
                    (float) grnd.Sample();// -0.01f;
                //(float)Math.Abs(grnd.Sample());
                //(float)(rnd.NextDouble() * 0.01 - 0.005);
            }
            for (var hi = 0; hi < HiddenNeuronCount; hi++)
            {
                this.HiddenNeuronBias[hi] =
                    //2.5f;
                        0f;
                    //-0.01f;
                    //0.2f;
                    //(float)grnd.Sample();
                    //(float)Math.Abs(grnd.Sample());
            }
            for (var hi = 0; hi < HiddenNeuronCount; hi++)
            {
                for (var vi = 0; vi < VisibleNeuronCount; vi++)
                {
                    this.Weights[hi][vi] =
                        (float)grnd.Sample();
                    //(float)Math.Abs(grnd.Sample());
                    //(float)(rnd.NextDouble() * 0.01 - 0.005);
                }
            }

            //имя машины
            this.Name = "RBM" + DateTime.Now.ToString("yyyyMMddHHmmss");
        }

        public void Train(
            string rootFolder,
            IDataSet trainData,
            IDataSet validationData,
            int maxGibbsChainLength,
            float learningRage,
            int epocheCount,
            IImageReconstructor imageReconstructor,
            int reconstructedCount,
            bool allowToSaveWeights,
            bool allowToSaveFeatures)
        {
            //создаем папку обучения
            var rbmFolder = string.IsNullOrEmpty(rootFolder) ? this.Name : rootFolder;
            Directory.CreateDirectory(rbmFolder);

            var hidden0NeuronSamples = new float[HiddenNeuronCount];
            var hidden1NeuronSamples = new float[HiddenNeuronCount];
            var visibleNeuronValues = new float[VisibleNeuronCount];

            for (var epocheNumber = 0; epocheNumber < epocheCount; epocheNumber++)
            {
                //создаем папку эпохи
                var epocheFolder = Path.Combine(rbmFolder, "epoche " + epocheNumber);
                Directory.CreateDirectory(epocheFolder);

                var before = DateTime.Now;

                var epochError = 0f;

                var inEpocheIndex = 0;
                foreach (var dataItem in trainData.CreateShuffledDataSet(_randomizer))
                {
                    #region progress report

                    if (inEpocheIndex % 50 == 0)
                    {
                        Console.Write(
                            "progress {0} out of {1}      ",
                            inEpocheIndex,
                            trainData.Count);
                        Console.SetCursorPosition(0, Console.CursorTop);
                    }

                    #endregion

                    //sample hidden
                    SampleHiddenLayer(
                        dataItem.Input,
                        hidden0NeuronSamples);

                    //cd-N

                    for (var cd = 0; cd < maxGibbsChainLength; cd++)
                    {
                        //sample visible
                        ComputeVisibleLayer(
                            cd == 0 ? hidden0NeuronSamples : hidden1NeuronSamples,
                            visibleNeuronValues);

                        //sample hidden again
                        SampleHiddenLayer(
                            visibleNeuronValues, 
                            hidden1NeuronSamples);
                    }

                    //end of cd-N

                    //change weights and biases
                    Parallel.For(0, HiddenNeuronCount, hi => 
                    //for (var hi = 0; hi < _hiddenNeuronCount; hi++)
                    {
                        for (var vi = 0; vi < VisibleNeuronCount; vi++)
                        {
                            float cd =
                                dataItem.Input[vi] * hidden0NeuronSamples[hi]
                                - visibleNeuronValues[vi] * hidden1NeuronSamples[hi];

                            Weights[hi][vi] += cd * learningRage;
                        }
                    }
                    );// Parallel.For

                    for (var hi = 0; hi < HiddenNeuronCount; hi++)
                    {
                        HiddenNeuronBias[hi] += (hidden0NeuronSamples[hi] - hidden1NeuronSamples[hi]) * learningRage;
                    }

                    for (var vi = 0; vi < VisibleNeuronCount; vi++)
                    {
                        VisibleNeuronBias[vi] += (dataItem.Input[vi] - visibleNeuronValues[vi]) * learningRage;
                    }

                    //-----------------------------------

                    //var havg = HiddenNeuronBias.Sum()/HiddenNeuronBias.Length;
                    //for (var hi = 0; hi < HiddenNeuronCount; hi++)
                    //{
                    //    HiddenNeuronBias[hi] -= havg;
                    //}

                    //var vavg = VisibleNeuronBias.Sum() / VisibleNeuronBias.Length;
                    //for (var vi = 0; vi < VisibleNeuronCount; vi++)
                    //{
                    //    VisibleNeuronBias[vi] -= vavg;
                    //}

                    //----------------------------------

                    inEpocheIndex++;
                }

                var after = DateTime.Now;
                Console.WriteLine("EpocheNumber takes {0}", (after - before));

                #region считаем ошибку и сохраняем реконструкцию, если задан реконструктор

                //var bitmap = new Bitmap(_imageWidth * 2 + 1, _imageHeight * reconstructedCount);
                var rIndex = 0;
                foreach (var dataItem in validationData)
                {
                    var reconstructed = this.GetReconstructed(
                        dataItem.Input,
                        true);

                    if (rIndex < reconstructedCount)
                    {
                        if (imageReconstructor != null)
                        {
                            imageReconstructor.AddPair(
                                rIndex,
                                reconstructed);
                            //CreateContrastEnhancedBitmapFrom(
                            //    bitmap, 
                            //    0, 
                            //    _imageHeight * rIndex,
                            //    _imageWidth,
                            //    _imageHeight,
                            //    dataItem.Input);

                            //CreateContrastEnhancedBitmapFrom(
                            //    bitmap, 
                            //    _imageWidth + 1, 
                            //    _imageHeight * rIndex,
                            //    _imageWidth,
                            //    _imageHeight,
                            //    reconstructed);
                        }
                    }

                    var sqdiff = 0.0f;
                    for (var cc = 0; cc < VisibleNeuronCount; cc++)
                    {
                        var dln = (reconstructed[cc] - dataItem.Input[cc]);
                        sqdiff += (float)Math.Sqrt(dln * dln);
                    }
                    epochError += sqdiff;

                    rIndex++;
                }
                if (imageReconstructor != null)
                {
                    imageReconstructor.GetReconstructedBitmap().Save(
                        Path.Combine(
                            epocheFolder,
                            "reconstruct.bmp"));
                    //bitmap.Save(
                    //    Path.Combine(
                    //        epocheFolder,
                    //        "reconstruct.bmp"));
                }

                #endregion

                #region сохраняем веса в виде файла

                this.SaveWeights(
                    Path.Combine(
                        epocheFolder,
                        "weights.bin"));

                #endregion

                #region сохраняем веса в виде изображения

                if (allowToSaveWeights)
                {
                    var totalFilterCount = Math.Min(500, HiddenNeuronCount);
                    var q0 = (int) Math.Ceiling(Math.Sqrt(totalFilterCount));
                    var wbmp = new Bitmap(_imageWidth*q0, _imageHeight*q0);
                    for (var hi = 0; hi < Math.Min(totalFilterCount, q0*q0); hi++)
                    {
                        CreateContrastEnhancedBitmapFrom(
                            wbmp,
                            (hi%q0)*_imageWidth,
                            ((int) (hi/q0))*_imageHeight,
                            _imageWidth,
                            _imageHeight,
                            this.Weights[hi].ToList().ConvertAll(Math.Abs).ToArray());
                    }
                    wbmp.Save(
                        Path.Combine(
                            epocheFolder,
                            "weights.bmp"));
                }

                #endregion

                #region сохраняем фичи

                if (allowToSaveFeatures)
                {
                    var totalFeaturesCount = Math.Min(500, HiddenNeuronCount);
                    var q1 = (int) Math.Ceiling(Math.Sqrt(totalFeaturesCount));
                    var fbmp = new Bitmap(_imageWidth*q1, _imageHeight*q1);
                    var ha = new float[HiddenNeuronCount];
                    for (var hi = 0; hi < Math.Min(totalFeaturesCount, q1 * q1); hi++)
                    {
                        ha[hi] = 1f;

                        var a = this.GetVisibleByHidden(ha);

                        CreateContrastEnhancedBitmapFrom(
                            fbmp,
                            (hi%q1)*_imageWidth,
                            ((int) (hi/q1))*_imageHeight,
                            _imageWidth,
                            _imageHeight,
                            a.ToList().ConvertAll(Math.Abs).ToArray());

                        ha[hi] = 0f;
                    }
                    fbmp.Save(
                        Path.Combine(
                            epocheFolder,
                            "feature.bmp"));
                }

                #endregion

                Console.WriteLine("EpocheNumber {0:D3}, per-item error {1}, total error {2}  ", epocheNumber, epochError / validationData.Count, epochError);
            }

            this.SaveStateHistogram(
                Path.Combine(
                    rbmFolder,
                    "weights_hist.csv"),
                Path.Combine(
                    rbmFolder,
                    "biases_hist.csv"));
        }


        #region reconstrution

        private void CreateContrastEnhancedBitmapFrom(
            Bitmap bitmap,
            int left,
            int top,
            int imageWidth,
            int imageHeight,
            float[] layer)
        {
            var max = layer.Take(imageWidth * imageHeight).Max(val => val);
            var min = layer.Take(imageWidth * imageHeight).Min(val => val);

            if (Math.Abs(min - max) <= float.Epsilon)
            {
                min = 0;
                max = 1;
            }

            for (int x = 0; x < imageWidth; x++)
            {
                for (int y = 0; y < imageHeight; y++)
                {
                    var value = layer[PointToIndex(x, y, imageHeight)];
                    value = (value - min) / (max - min);
                    var b = (byte)Math.Max(0, Math.Min(255, value * 255.0));

                    bitmap.SetPixel(left + x, top + y, Color.FromArgb(b, b, b));
                }
            }
        }

        private int PointToIndex(int x, int y, int width)
        {
            return y * width + x;
        }

        private float[] GetVisibleByHidden(
            float[] hiddenNeuronValues,
            bool useBias = true)
        {
            var visibleNeuronValues = new float[this.VisibleNeuronCount];

            //compute visible
            for (var vi = 0; vi < VisibleNeuronCount; vi++)
            {
                //считаем значение нейрона
                var visibleNeuronValue = CalculateVisibleNeuron(vi, hiddenNeuronValues, useBias);

                visibleNeuronValues[vi] = visibleNeuronValue;
            }

            return visibleNeuronValues;
        }

        private float[] GetReconstructed(
            float[] input,
            bool useBias = true)
        {
            var hiddenNeuronValues = new float[this.HiddenNeuronCount];
            var visibleNeuronValues = new float[this.VisibleNeuronCount];

            //compute hidden
            this.SampleHiddenLayer(
                input,
                hiddenNeuronValues,
                useBias,
                false);
            //for (var hi = 0; hi < _hiddenNeuronCount; hi++)
            //{
            //    //считаем значение нейрона
            //    var hiddenNeuronValue = CalculateHiddenNeuron(hi, input, useBias);

            //    //не сэмплируем значение нейрона
            //    hiddenNeuronValues[hi] = hiddenNeuronValue;
            //}

            //compute visible
            this.ComputeVisibleLayer(
                hiddenNeuronValues,
                visibleNeuronValues,
                useBias);
            //for (var vi = 0; vi < _visibleNeuronCount; vi++)
            //{
            //    //считаем значение нейрона
            //    var visibleNeuronValue = CalculateVisibleNeuron(vi, hiddenNeuronValues, useBias);

            //    visibleNeuronValues[vi] = visibleNeuronValue;
            //}

            return visibleNeuronValues;
        }

        #endregion

        #region calculates and samples

        public void SampleHiddenLayer(
            float[] visibleNeuronValues,
            float[] hiddenNeuronSamples,
            bool useBias = true,
            bool needToSample = true)
        {
            for (var hi = 0; hi < HiddenNeuronCount; hi++)
            {
                //считаем значение нейрона
                var hiddenNeuronValue = CalculateHiddenNeuron(
                    hi,
                    visibleNeuronValues,
                    useBias);

                //сэмплируем значение нейрона
                var hiddenNeuronSample =
                    needToSample
                        ? SampleHiddenNeuronWithNRelu(hiddenNeuronValue)
                        : hiddenNeuronValue;

                hiddenNeuronSamples[hi] = hiddenNeuronSample;
            }
        }

        public void ComputeVisibleLayer(
            float[] hiddenNeuronSamples,
            float[] visibleNeuronValues,
            bool useBias = true)
        {
            for (var vi = 0; vi < VisibleNeuronCount; vi++)
            {
                //считаем значение нейрона
                var visibleNeuronValue = CalculateVisibleNeuron(
                    vi,
                    hiddenNeuronSamples,
                    useBias);

                //значение нейрона не семплируем
                visibleNeuronValues[vi] = visibleNeuronValue;
            }
        }

        private float CalculateVisibleNeuron(
            int vi,
            float[] hiddenStates,
            bool useBias = true)
        {
            var visibleNeuronValue = useBias ? VisibleNeuronBias[vi] : 0f;

            for (var hi = 0; hi < HiddenNeuronCount; hi++)
            {
                visibleNeuronValue += Weights[hi][vi] * hiddenStates[hi];
            }

            return visibleNeuronValue;
        }

        private float CalculateHiddenNeuron(
            int hi,
            float[] visibleStates,
            bool useBias = true)
        {
            var hiddenNeuronValue = useBias ? HiddenNeuronBias[hi] : 0f;

            for (var vi = 0; vi < VisibleNeuronCount; vi++)
            {
                hiddenNeuronValue += Weights[hi][vi] * visibleStates[vi];
            }

            return hiddenNeuronValue;
        }

        private float SampleVisibleNeuronWithLinear(float x)
        {
            return
                x;
        }

        private float SampleHiddenNeuronWithNRelu(float x)
        {
            var stdDev = ComputeSigmoid(x);

            _gaussRandom.StdDev = stdDev;

            var normalNoise = (float)_gaussRandom.Sample();

            return
                Math.Max(0f, x + normalNoise);
        }

        public float ComputeSigmoid(float x)
        {
            var r = (float)(1.0 / (1.0 + Math.Exp(-x)));
            return r;
        }

        #endregion

        #region histogram

        public void SaveStateHistogram(string weightFilename, string biasFilename)
        {
            this.SaveWeightHistogram(weightFilename);
            this.SaveBiasHistogram(biasFilename);
        }

        public void SaveBiasHistogram(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            var discreteBiases =
                from b in this.VisibleNeuronBias.Concat(this.HiddenNeuronBias)
                select ((int)(b * 50)) / 50f;

            var countw =
                from b in discreteBiases
                group b by b
                    into bgroup
                    orderby bgroup.Key ascending
                    select bgroup;

            //вставляем строку (с помощью нее легче в екселе построить правильную гистограмму)
            File.AppendAllText(
                filename,
                ";Biases Count\r\n");

            File.AppendAllLines(
                filename,
                countw.ToList().ConvertAll(j => j.Key.ToString() + ";" + j.Count().ToString()));
        }


        public void SaveWeightHistogram(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            var discreteWeights =
                from w in this.Weights
                from ww in w
                select ((int)(ww * 50)) / 50f;

            var countw =
                from w in discreteWeights
                group w by w
                    into wgroup
                    orderby wgroup.Key ascending
                    select wgroup;

            //вставляем строку (с помощью нее легче в екселе построить правильную гистограмму)
            File.AppendAllText(
                filename,
                ";Weight Count\r\n");

            File.AppendAllLines(
                filename,
                countw.ToList().ConvertAll(j => j.Key.ToString() + ";" + j.Count().ToString()));
        }

        #endregion

        #region save-load state

        public void SaveWeights(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            var d = new float[(this.HiddenNeuronCount + 1) * (this.VisibleNeuronCount + 1)];

            var toIndex = 0;

            //хидден веса + биасы
            for (var hiddenIndex = 0; hiddenIndex < this.HiddenNeuronCount; hiddenIndex++)
            {
                Array.Copy(
                    this.Weights[hiddenIndex],
                    0,
                    d,
                    toIndex,
                    this.VisibleNeuronCount);
                d[toIndex + this.VisibleNeuronCount] = this.HiddenNeuronBias[hiddenIndex];

                toIndex += this.VisibleNeuronCount + 1;
            }

            //теперь визибл биасы
            for (var visibleIndex = 0; visibleIndex < this.VisibleNeuronCount; visibleIndex++)
            {
                d[toIndex + visibleIndex] = this.VisibleNeuronBias[visibleIndex];
            }

            new SerializationHelper().SaveToFile(d, filename);
        }

        public void LoadWeights(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            var d = new SerializationHelper().LoadFromFile<float[]>(filename);

            //хидден веса + биасы
            var toIndex = 0;
            for (var hiddenIndex = 0; hiddenIndex < this.HiddenNeuronCount; hiddenIndex++)
            {
                Array.Copy(
                    d,
                    toIndex,
                    this.Weights[hiddenIndex],
                    0,
                    this.VisibleNeuronCount);
                this.HiddenNeuronBias[hiddenIndex] = d[toIndex + this.VisibleNeuronCount];

                toIndex += this.VisibleNeuronCount + 1;
            }

            //теперь визибл биасы
            for (var visibleIndex = 0; visibleIndex < this.VisibleNeuronCount; visibleIndex++)
            {
                this.VisibleNeuronBias[visibleIndex] = d[toIndex + visibleIndex];
            }

        }

        #endregion

    }
}
