using System;
using MyNN.Data;

namespace MyNN.KNN
{
    public class CPUOpenCLKNearestFactory : IKNearestFactory
    {
        public IKNearest CreateKNearest(IDataSet dataList)
        {
            if (dataList == null)
            {
                throw new ArgumentNullException("dataList");
            }

            return 
                new CPUOpenCLKNearest(dataList);
        }
    }
}