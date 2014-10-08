using System;
using MyNN.Common.Data;

namespace MyNN.KNN.OpenCL.CPU.Factory
{
    public class KNearestFactory : IKNearestFactory
    {
        public IKNearest CreateKNearest(IDataSet dataList)
        {
            if (dataList == null)
            {
                throw new ArgumentNullException("dataList");
            }

            return 
                new KNearest(dataList);
        }
    }
}