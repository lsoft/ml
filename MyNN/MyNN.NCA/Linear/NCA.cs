using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Generic;
using MathNet.Numerics.LinearAlgebra.Single;
using MyNN.Common.Data;


namespace MyNN.NCA.Linear
{
    #region support classes

    #endregion

    public class NCA
    {
        private int _itemSize;

        private readonly List<DataVector> _trainList;

        private readonly Random _rnd;
        
        public DenseMatrix A
        {
            get;
            private set;
        }

        public NCA()
        {
            //MathNet.Numerics.Control.LinearAlgebraProvider = new AtlasLinearAlgebraProvider();

            _trainList = new List<DataVector>();

            _rnd = new Random(1); //!!!
        }

        public void Train(
            int aMatrixRowCount,
            List<IDataItem> trainData,
            float learningRate,
            int epocheCount,
            int batchSize)
        {
            if (trainData == null || trainData.Count == 0)
            {
                throw new ArgumentNullException("trainData");
            }

            Console.WriteLine("NCA train start...");

            _itemSize = trainData[0].Input.Length;

            //приводим данные к удобному виду
            foreach (var tItem in trainData)
            {
                var classId = tItem.Output.ToList().FindIndex(j => j > float.Epsilon);

                var vector = new DataVector(classId, new DenseVector(tItem.Input));

                this._trainList.Add(vector);
            }

            //формируем матрицу со значениями по умолчанию
            var adata = new float[aMatrixRowCount * _itemSize];
            for (var cc = 0; cc < adata.Length; cc++)
            {
                adata[cc] = (float)_rnd.NextDouble() * 2.0f - 1.0f;
            }
            A = new DenseMatrix(aMatrixRowCount, _itemSize, adata);

            #region dump

            File.Delete("_pij.csv");
            File.Delete("_pi.csv");

            //записываем необработанные данные в файл
            var fileContent0 = new List<string>();
            foreach (var t in _trainList)
            {
                var x = t.Data;
                fileContent0.Add(
                    string.Join(";", x.Values.ToList().ConvertAll(k => k.ToString()))
                     + ";" + t.ClassId);
            }

            File.Delete("_" + (0) + ".csv");
            File.WriteAllLines("_" + (0) + ".csv", fileContent0.ToArray());

            File.Delete("_sum.csv");

            #endregion

            for (var epocheIndex = 0; epocheIndex < epocheCount; epocheIndex++)
            {
                var batchIndex = 0;
                while (true)
                {
                    #region

                    var x0 = DateTime.Now;

                    #endregion

                    var pijCalculator = new PijCalculator(A, this._trainList);

                    Matrix<float> summ = new DenseMatrix(this._itemSize, this._itemSize);
                    var i = 0;
                    for (i = batchIndex * batchSize; i < Math.Min((batchIndex + 1) * batchSize, _trainList.Count); i++)
                    {
                        pijCalculator.PrepareForI(i);

                        var ci = _trainList[i].ClassId;
                        var xi = _trainList[i].Data;

                        Matrix<float> part1 = new DenseMatrix(this._itemSize, this._itemSize, 0f);
                        Matrix<float> part2 = new DenseMatrix(this._itemSize, this._itemSize, 0f);
                        float pi = 0.0f;

                        for (var k = 0; k < _trainList.Count; k++)
                        {
                            var pik = pijCalculator.GetPij(i, k);
                            if (pik <= -float.Epsilon || pik >= float.Epsilon) //оптимизация
                            {
                                var itemK = _trainList[k];

                                var xk = itemK.Data;
                                var xik = xi - xk;
                                var xikT = xik.ToRowMatrix();

                                var iter = pik * xik.ToColumnMatrix() * xikT;

                                part1 += iter;

                                if (itemK.ClassId == ci)
                                {
                                    part2 += iter;
                                    pi += pik;
                                }
                            }
                        }

                        var diff = pi * part1 - part2;
                        summ += diff;
                    }

                    #region dump

                    //if (epocheIndex % 10 == 0)
                    {
                        pijCalculator.Dump();
                    }

                    #endregion

                    var dFdA = (DenseMatrix)((learningRate * 2.0f) * A * summ);

                    #region dump

                    //if (epocheIndex % 10 == 0)
                    {
                        var x1 = DateTime.Now;
                        var timeDiff = x1 - x0;

                        var sum = dFdA.Values.Sum(j => Math.Abs(j));
                        File.AppendAllText("_sum.csv", sum.ToString() + "\r\n");

                    }
                    #endregion

                    A += dFdA;

                    #region dump

                    //if (epocheIndex % 10 == 0)
                    {
                        //записываем обработанные данные в файл
                        var fileContent = new List<string>();
                        foreach (var t in _trainList)
                        {
                            var x = A * t.Data;
                            fileContent.Add(
                                string.Join(
                                    ";",
                                    x.Values.ToList().ConvertAll(k => k.ToString()))
                                + ";" + t.ClassId);
                        }

                        File.Delete("_" + (epocheIndex + 1) + ".csv");
                        File.WriteAllLines("_" + (epocheIndex + 1) + ".csv", fileContent.ToArray());

                    }

                    #endregion

                    batchIndex++;

                    if (i == _trainList.Count)
                    {
                        break;
                    }
                }

                Console.WriteLine("NCA epoche finished...");
            }

            #region dump a

            File.Delete("_a.csv");

            for (var row = 0; row < A.RowCount; row++)
            {
                var s = string.Join(";", A.Row(row).ToList().ConvertAll(j => j.ToString()));
                File.AppendAllText("_a.csv", s + "\r\n");
            }

            #endregion

            Console.WriteLine("NCA train finished...");

        }
    }
}
