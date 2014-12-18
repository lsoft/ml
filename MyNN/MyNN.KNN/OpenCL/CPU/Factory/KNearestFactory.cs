using System;
using System.Collections.Generic;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;

namespace MyNN.KNN.OpenCL.CPU.Factory
{
    public class KNearestFactory : IKNearestFactory
    {
        public IKNearest CreateKNearest(
            IList<IDataItem> dataList
            )
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