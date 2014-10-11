using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Other;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.KNN.OpenCL.CPU
{
    public class KNearest : IKNearest
    {
        private readonly CLProvider _clProvider;

        private readonly IDataSet _dataList;
        private readonly int _dataCount;
        private readonly int _coordinateCount;

        private readonly MemFloat _dataMem;
        private readonly MemFloat _resultMem;
        private readonly MemFloat _itemMem;

        private readonly Kernel _calculateDistanceKernel;


        public KNearest(IDataSet dataList)
        {
            #region validate

            if (dataList == null)
            {
                throw new ArgumentNullException("dataList");
            }
            if (dataList.Count == 0)
            {
                throw new ArgumentException("dataList");
            }

            #endregion

            _clProvider = new CLProvider();

            _dataList = dataList;
            _dataCount = dataList.Count;
            _coordinateCount = dataList[0].Input.Length;

            //создаем массивы данных
            _dataMem = _clProvider.CreateFloatMem(
                _dataCount * _coordinateCount,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);
            _itemMem = _clProvider.CreateFloatMem(
                _coordinateCount,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);
            _resultMem = _clProvider.CreateFloatMem(
                _dataCount,
                MemFlags.CopyHostPtr | MemFlags.WriteOnly);

            this._kernelSource = this._kernelSource.Replace(
                "{0}",
                _coordinateCount.ToString());

            //создаем кернелы
            _calculateDistanceKernel = _clProvider.CreateKernel(
                _kernelSource,
                "CalculateDistanceKernel");

            //заполняем массивы данных
            var index = 0;
            foreach (var i in dataList)
            {
                Array.Copy(i.Input, 0, _dataMem.Array, index, _coordinateCount);

                index += _coordinateCount;
            }
            _dataMem.Write(BlockModeEnum.Blocking);
        }

        public int Classify(
            float[] itemToClassify,
            int knn)
        {
            #region validate

            if (itemToClassify == null)
            {
                throw new ArgumentNullException("itemToClassify");
            }
            if (itemToClassify.Length != _coordinateCount)
            {
                throw new ArgumentException("itemToClassify");
            }
            if (knn <= 0 || knn % 2 == 0)
            {
                throw new ArgumentException("knn");
            }

            #endregion

            //очищаем результаты
            Array.Clear(_resultMem.Array, 0, _resultMem.Array.Length);
            _resultMem.Write(BlockModeEnum.Blocking);

            //заполняем проверочную точку
            Array.Copy(itemToClassify, 0, _itemMem.Array, 0, itemToClassify.Length);
            _itemMem.Write(BlockModeEnum.Blocking);

            //запускаем кернел
            _calculateDistanceKernel
                .SetKernelArgMem(0, _dataMem)
                .SetKernelArgMem(1, _itemMem)
                .SetKernelArgMem(2, _resultMem)
                .SetKernelArg(3, 4, _coordinateCount)
                .EnqueueNDRangeKernel(_dataCount);

            _clProvider.QueueFinish();

            _resultMem.Read(BlockModeEnum.Blocking);

            //добавляем в массив индекс
            var xlist = new List<Pair<int, float>>();
            for (var cc = 0; cc < _resultMem.Array.Length; cc++)
            {
                xlist.Add(
                    new Pair<int, float>(cc, _resultMem.Array[cc]));
            }

            var result = -1;

            //сортируем результаты по увеличению удаления
            var sortedDistanceList =
                (from y in xlist
                 orderby y.Second ascending
                 select y.First);
            var enumerator = sortedDistanceList.GetEnumerator();
            
            //извлекаем классы
            if (enumerator.MoveNext())
            {
                var indexes = new List<int>();
                for (var cc = 0; cc < knn; cc++)
                {
                    var indexMinDistance = enumerator.Current;
                    indexes.Add(indexMinDistance);

                    if (!enumerator.MoveNext())
                    {
                        break;
                    }
                }

                //выбираем один класс
                if (indexes.Count > 0)
                {
                    var a =
                        (from y in indexes
                         orderby indexes.Count(k => _dataList[y].OutputIndex == _dataList[k].OutputIndex) descending
                         select _dataList[y].OutputIndex).First();

                    result = a;
                }
            }

            return result;
        }

        private string _kernelSource = @"
typedef struct
{
    float Coordinates[{0}];
} Item;

__kernel void CalculateDistanceKernel(
    __global Item * items,
    __global Item * checkedItem,
    __global float * distance,
    int coordinateCount)
{
    int itemIndex = get_global_id(0);

    Item processingItem = items[itemIndex];

    int index = 0;
    float4 sum4 = 0;
    for (index = 0; index < coordinateCount / 4; index++)
    {
        float4 c4 = vload4(index, processingItem.Coordinates);
        float4 d4 = vload4(index, checkedItem->Coordinates);
        float4 diff4 = c4 - d4;

        sum4 += diff4 * diff4;
    }
    float sum = sum4.s0 + sum4.s1 + sum4.s2 + sum4.s3;
    for (index = index * 4; index < coordinateCount; index++)
    {
        float c = processingItem.Coordinates[index];
        float d = checkedItem->Coordinates[index];
        float diff = c - d;

        sum += diff * diff;
    }

    distance[itemIndex] = sum;
}

";
    }
}
