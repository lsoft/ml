using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data.Visualizer;
using MyNN.MLP2.Randomizer;

namespace MyNN.Data
{
    [Serializable]
    public class DataSet : IEnumerable<DataItem>, IVisualizer
    {
        public IVisualizer Visualizer
        {
            get;
            private set;
        }

        /// <summary>
        /// Данный датасет способен визуализировать информацию
        /// </summary>
        public bool IsAbleToVisualize
        {
            get
            {
                return
                    Visualizer != null;
            }
        }

        public List<DataItem> Data
        {
            get;
            private set;
        }

        public bool IsAuencoderDataSet
        {
            get;
            private set;
        }

        public bool IsClassificationAuencoderDataSet
        {
            get;
            private set;
        }

        public int Count
        {
            get
            {
                return
                    this.Data.Count;
            }
        }

        public DataSet(
            IVisualizer visualizer = null)
        {
            Visualizer = visualizer;
            Data = new List<DataItem>();

            IsAuencoderDataSet = false;
            IsClassificationAuencoderDataSet = false;
        }

        public DataSet(
            List<DataItem> data,
            IVisualizer visualizer = null)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            Data = data;
            Visualizer = visualizer;

            IsAuencoderDataSet = false;
            IsClassificationAuencoderDataSet = false;
        }

        public DataSet(DataSet dataSet)
        {
            #region validate

            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            #endregion

            Visualizer = dataSet.Visualizer;
            Data = new List<DataItem>(dataSet.Data);

            IsAuencoderDataSet = false;
            IsClassificationAuencoderDataSet = false;
        }

        public DataSet(DataSet dataSet, int takeCount)
        {
            #region validate

            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            if (takeCount <= 0)
            {
                throw new ArgumentException("takeCount <= 0", "takeCount");
            }

            #endregion

            Visualizer = dataSet.Visualizer;
            Data = new List<DataItem>(dataSet.Data.Take(takeCount));

            IsAuencoderDataSet = false;
            IsClassificationAuencoderDataSet = false;
        }

        public DataSet(
            DataSet dataSet,
            List<float[]> inputPart,
            IVisualizer visualizer)
        {
            //visualizer allowed to be null

            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }
            if (inputPart == null)
            {
                throw new ArgumentNullException("inputPart");
            }
            if (dataSet.Count != inputPart.Count)
            {
                throw new InvalidOperationException("Датасет и данные должны быть одного размера");
            }

            Visualizer = visualizer;
            this.Data = new List<DataItem>();

            for (var i = 0; i < dataSet.Count; i++)
            {
                var di = new DataItem(inputPart[i], dataSet[i].Output);
                this.Data.Add(di);
            }

            IsAuencoderDataSet = false;
            IsClassificationAuencoderDataSet = false;
        }

        public void AddItem(DataItem di)
        {
            this.Data.Add(di);
        }

        public DataItem this[int i]
        {
            get
            {
                return
                    this.Data[i];
            }
        }

        public DataSet ConvertToClassificationAutoencoder()
        {
            var result =
                new DataSet(
                    this.Data.ConvertAll(j => new DataItem(j.Input, j.Output.Concatenate(j.Input))),
                    this.Visualizer);

            result.IsClassificationAuencoderDataSet = true;

            return result;
        }

        public DataSet ConvertToAutoencoder()
        {
            var result =
                new DataSet(
                    this.Data.ConvertAll(j => new DataItem(j.Input, j.Input)),
                    this.Visualizer);

            result.IsAuencoderDataSet = true;

            return result;
        }

        public List<float[]> GetInputPart()
        {
            return
                this.Data.ConvertAll(j => j.Input);
        }

        public void ExpandDataSet(
            IRandomizer randomizer,
            float scale,
            int epocheCount)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            Console.WriteLine("Generate expanded set...");

            var resultData = new List<DataItem>();

            for (var epocheNumber = 0; epocheNumber < epocheCount; epocheNumber++)
            {
                Console.WriteLine("Generate expanded set, epoche " + epocheNumber);

                foreach (var d in this.Data)
                {
                    var inputd = new float[d.Input.Length];
                    Array.Copy(d.Input, inputd, inputd.Length);

                    var outputd = new float[d.Output.Length];
                    Array.Copy(d.Output, outputd, outputd.Length);

                    for (var cc = 0; cc < inputd.Length; cc++)
                    {
                        var linearcoef = (1.0f - (randomizer.Next() * scale / 2.0f - scale));
                        inputd[cc] *= linearcoef;

                        var sincoef = (float)(Math.Sin(cc * 0.01f) * scale + 1.0f);
                        inputd[cc] *= sincoef;
                    }

                    resultData.Add(
                        new DataItem(inputd, outputd));
                }
            }

            this.Data.AddRange(resultData);
        }

        /// <summary>
        /// Линейная нормализация [0;1]
        /// </summary>
        public void Normalize(float bias = 0f)
        {
            for (var cc = 0; cc < this.Data.Count; cc++)
            {
                var data = this.Data[cc].Input;

                var min = data.Min();
                var max = data.Max();

                for (var dd = 0; dd < data.Length; dd++)
                {
                    var i = data[dd];
                    i -= min;
                    i /= (-min + max);
                    data[dd] = i - bias;
                }
            }

            #region validate

            var min2 = this.Data
                           .ConvertAll(j => j.Input)
                           .Min(j => j.Min());

            if (Math.Abs(min2) > bias)
            {
                throw new InvalidOperationException("min2");
            }

            var max2 = this.Data
                           .ConvertAll(j => j.Input)
                           .Max(j => j.Max());

            if (Math.Abs(max2 - 1.0f) > bias)
            {
                throw new InvalidOperationException("max2");
            }

            #endregion
        }

        /// <summary>
        /// Гауссова нормализация
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        public void GNormalize()
        {
            var index = 0;
            foreach (var item in this.Data)
            {
                var input = item.Input;

                var mean0 =
                    (float)MathNet.Numerics.Statistics.Statistics.Mean(input.ToList().ConvertAll(j => (double)j));

                var variance0 =
                    (float)MathNet.Numerics.Statistics.Statistics.Variance(input.ToList().ConvertAll(j => (double)j));

                var standardDeviation0 =
                    (float)MathNet.Numerics.Statistics.Statistics.StandardDeviation(input.ToList().ConvertAll(j => (double)j));

                //приводим к среднему = 0 и дисперсии = 1
                for (var i = 0; i < input.Length; i++)
                {
                    input[i] -= mean0;

                    input[i] /= (float)Math.Sqrt(variance0);
                }

                if (index++ == 0)
                {
                    var mean1 =
                        (float)MathNet.Numerics.Statistics.Statistics.Mean(input.ToList().ConvertAll(j => (double)j));

                    var variance1 =
                        (float)MathNet.Numerics.Statistics.Statistics.Variance(input.ToList().ConvertAll(j => (double)j));

                    var standardDeviation1 =
                        (float)MathNet.Numerics.Statistics.Statistics.StandardDeviation(input.ToList().ConvertAll(j => (double)j));

                    Console.WriteLine("mean {0}   variance {1}   std {2}", mean0, variance0, standardDeviation0);
                    Console.WriteLine("mean {0}   variance {1}   std {2}", mean1, variance1, standardDeviation1);
                }
            }
        }

        /// <summary>
        /// Создает новый датасет, перемешивает его и отдает
        /// </summary>
        /// <returns></returns>
        public DataSet CreateShuffledDataSet(
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            var cloned = new List<DataItem>(this.Data);
            for (int i = 0; i < cloned.Count - 1; i++)
            {
                if (randomizer.Next() >= 0.5d)
                {
                    var newIndex = randomizer.Next(cloned.Count);

                    var tmp = cloned[i];
                    cloned[i] = cloned[newIndex];
                    cloned[newIndex] = tmp;
                }
            }

            return
                new DataSet(
                    cloned,
                    this.Visualizer);
        }

        /// <summary>
        /// Бинаризует данные в датасете
        /// (1 с вероятностью значения)
        /// Если данные не нормализованы в диапазон [0;1], генерируется исключение
        /// </summary>
        /// <returns></returns>
        public DataSet CreateBinarizedDataSet(
            IRandomizer randomizer)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            var cloned = new List<DataItem>();
            foreach (var di in this.Data)
            {
                if (di.Input.Any(j => j < 0f || j > 1f))
                {
                    throw new InvalidOperationException("Данные не нормализованы в диапазон [0;1]");
                }

                var bi = di.Input.ToList().ConvertAll(j => (randomizer.Next() < j) ? 1f : 0f);

                var ndi = new DataItem(bi.ToArray(), di.Output);
                cloned.Add(ndi);
            }

            return 
                new DataSet(
                    cloned, 
                    this.Visualizer);
        }


        #region get enumerator interface implementation

        public IEnumerator<DataItem> GetEnumerator()
        {
            return
                this.Data.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return
                GetEnumerator();
        }

        #endregion

        public void SaveAsGrid(string filepath, List<float[]> data)
        {
            if (filepath == null)
            {
                throw new ArgumentNullException("filepath");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            if (Visualizer != null)
            {
                Visualizer.SaveAsGrid(filepath, data);
            }
        }

        public void SaveAsPairList(string filepath, List<Pair<float[], float[]>> data)
        {
            if (filepath == null)
            {
                throw new ArgumentNullException("filepath");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            if (Visualizer != null)
            {
                Visualizer.SaveAsPairList(filepath, data);
            }
        }
    }
}