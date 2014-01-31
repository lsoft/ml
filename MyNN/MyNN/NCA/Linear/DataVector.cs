using System;
using MathNet.Numerics.LinearAlgebra.Single;

namespace MyNN.NCA.Linear
{
    public class DataVector
    {
        public int ClassId
        {
            get;
            private set;
        }

        public DenseVector Data
        {
            get;
            private set;
        }

        public DataVector(int classId, DenseVector data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            ClassId = classId;
            Data = data;
        }
    }
}