using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Matrix = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix;
using Vector = MathNet.Numerics.LinearAlgebra.Single.DenseVector;

namespace MyNN.NCA.Linear
{
    public class PijCalculator
    {
        private readonly Matrix _a;
        private readonly List<DataVector> _trainList;
        private Pair<int, float> _pairIZnamenatel;

        private readonly List<Vector> _axiList;
        private readonly Matrix _pij;

        public PijCalculator(
            Matrix a,
            List<DataVector> trainList)
        {
            #region validate

            if (a == null)
            {
                throw new ArgumentNullException("a");
            }
            if (trainList == null)
            {
                throw new ArgumentNullException("trainList");
            }

            #endregion

            _a = a;
            _trainList = trainList;
            
            _axiList = new List<Vector>();

            //вычисляем все _a * xi
            foreach (var item in trainList)
            {
                var axi = (_a*item.Data);
                _axiList.Add(axi);
            }

            _pij = new Matrix(trainList.Count, trainList.Count);
        }

        public void PrepareForI(int i)
        {
            //var xi = _trainList[i].Data;
            var axi = _axiList[i];

            var znamenatel = 0f;
            for (var k = 0; k < _trainList.Count; k++)
            {
                if (k != i)
                {
                    //var xk = _trainList[k].Data;
                    var axk = _axiList[k];

                    //var zn_x = _a * xi - _a * xk;
                    var zn_x = axi - axk;
                    var zn_distance = zn_x * zn_x;
                    var partZnamenatel = (float)Math.Exp(-zn_distance);

                    znamenatel += partZnamenatel;
                }
            }

            if (znamenatel < float.Epsilon)
            {
                throw new InvalidOperationException("znamenatel < float.Epsilon");
            }

            _pairIZnamenatel = new Pair<int, float>(i, znamenatel);
        }

        public float GetPij(int i, int j)
        {
            if (_pairIZnamenatel == null || _pairIZnamenatel.First != i)
            {
                throw new InvalidOperationException("_pairIZnamenatel == null || _pairIZnamenatel.First != i");
            }

            if (i != j)
            {
                //var xi = _trainList[i].Data;
                //var xj = _trainList[j].Data;
                var axi = _axiList[i];
                var axj = _axiList[j];

                #region числитель

                //var ch_x = _a * xi - _a * xj;
                var ch_x = axi - axj;
                var ch_distance = ch_x * ch_x;
                var chislitel = (float)Math.Exp(-ch_distance);

                #endregion

                _pij[i, j] = chislitel / _pairIZnamenatel.Second;
            }

            return _pij[i, j];
        }

        public void Dump()
        {
            DumpPij();
            DumpPi();
        }

        private void DumpPi()
        {
            File.AppendAllText("_pi.csv", "\r\n");

            var s = string.Empty;
            for (var row = 0; row < this._trainList.Count; row++)
            {
                var ci = this._trainList[row].ClassId;

                var classList = this._trainList.FindAll(j => j.ClassId == ci);

                var pi = classList.Sum(j => _pij[row, _trainList.IndexOf(j)]);

                s += pi + ";";
            }

            File.AppendAllText("_pi.csv", s);
        }

        private void DumpPij()
        {
            File.AppendAllText("_pij.csv", "\r\n");

            for (var row = 0; row < this._trainList.Count; row++)
            {
                var s = string.Join(";", this._pij.Row(row).ToList().ConvertAll(j => j.ToString()));
                File.AppendAllText("_pij.csv", s + "\r\n");
            }
        }
    }
}
