using System;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;

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