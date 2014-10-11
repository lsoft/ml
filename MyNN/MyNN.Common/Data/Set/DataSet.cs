using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Dense;

namespace MyNN.Common.Data.Set
{
    [Serializable]
    public class DataSet : IDataSet
    {
        public List<IDataItem> Data
        {
            get;
            private set;
        }

        public bool IsAutoencoderDataSet
        {
            get;
            set;
        }

        public bool IsAuencoderDataSet
        {
            get
            {
                var result = false;

                if (this.Data.Count > 0)
                {
                    result = this.Data[0].InputLength == this.Data[0].OutputLength;
                }

                return result;
            }
        }

        public int Count
        {
            get
            {
                return
                    this.Data.Count;
            }
        }

        public int InputLength
        {
            get
            {
                var result = 0;

                if (this.Data.Count > 0)
                {
                    result = this.Data[0].InputLength;
                }

                return result;
            }
        }

        public DataSet()
        {
            Data = new List<IDataItem>();
        }

        public DataSet(
            List<IDataItem> data
            )
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            Data = data;
        }


        public DataSet(IDataSet dataSet, int takeCount)
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

            Data = new List<IDataItem>(dataSet.Data.Take(takeCount));
        }

        public DataSet(
            IDataSet dataSet,
            List<float[]> inputPart,
            IDataItemFactory dataItemFactory
            )
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }
            if (inputPart == null)
            {
                throw new ArgumentNullException("inputPart");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (dataSet.Count != inputPart.Count)
            {
                throw new InvalidOperationException("ƒатасет и данные должны быть одного размера");
            }

            this.Data = new List<IDataItem>();

            for (var i = 0; i < dataSet.Count; i++)
            {
                var di = dataItemFactory.CreateDataItem(inputPart[i], dataSet[i].Output);
                this.Data.Add(di);
            }
        }

        public IDataItem this[int i]
        {
            get
            {
                return
                    this.Data[i];
            }
        }

        public List<float[]> GetInputPart()
        {
            return
                this.Data.ConvertAll(j => j.Input);
        }

        /// <summary>
        /// Ћинейна€ нормализаци€ [0;1]
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
        /// √ауссова нормализаци€
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


        #region get enumerator interface implementation

        public IEnumerator<IDataItem> GetEnumerator()
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

    }
}