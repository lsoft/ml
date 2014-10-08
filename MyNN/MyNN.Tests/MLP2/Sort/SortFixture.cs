using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4.Sorter;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.Sort
{
    [TestClass]
    public class SortFixture
    {
        [TestMethod]
        public void CSharpDefaultSortTest()
        {
            Func<List<SortItem>, List<SortItem>> sort =
                list =>
                {
                    return 
                        (from i in list
                         orderby ((long)i.AIndex << 32) + i.BIndex
                        select i).ToList();
                };

            TestSortAlgorithm(sort);
        }

        [TestMethod]
        public void OpenCLBitonicSortTest()
        {
            Func<List<SortItem>, List<SortItem>> sort =
                list =>
                {
                    const int structureSizeOf = (sizeof (uint)*2 + sizeof (float));

                    using (var clProvider = new CLProvider())
                    {
                        ulong totalElementCount = (ulong) list.Count;
                        double log2d = Math.Log(totalElementCount, 2);
                        uint log2 = ((log2d%1) > double.Epsilon) ? ((uint) log2d + 1) : (uint) log2d;
                        ulong totalElementCountPlusOverhead = (ulong)Math.Pow(2, log2);

                        //заполняем массив
                        var sourceMem = clProvider.CreateByteMem(
                            totalElementCountPlusOverhead * structureSizeOf,
                            MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                        
                        //заполняем его значением 255 (это максимальный SortItem.Key)
                        //заполнять необходимо, так как есть оверхед, чтобы алгоритм сортировки работал правильно
                        sourceMem.Array.Fill((byte)255);

                        var index = 0;
                        foreach (var i in list)
                        {
                            var aBytes = BitConverter.GetBytes(i.AIndex);
                            aBytes.CopyTo(sourceMem.Array, index * 12 + 0);

                            var bBytes = BitConverter.GetBytes(i.BIndex);
                            bBytes.CopyTo(sourceMem.Array, index * 12 + 4);

                            var dBytes = BitConverter.GetBytes(i.Distance);
                            dBytes.CopyTo(sourceMem.Array, index * 12 + 8);

                            index++;
                        }

                        sourceMem.Write(BlockModeEnum.Blocking);

                        var sorter = new AMDBitonicSorter(clProvider);
                        sorter.Sort(
                            sourceMem,
                            totalElementCountPlusOverhead);

                        //читаем данные
                        sourceMem.Read(BlockModeEnum.Blocking);

                        var tmpArray = new byte[structureSizeOf];
                        var result = new List<SortItem>();
                        for (ulong cc = 0; cc < totalElementCount; cc++)
                        {
                            Array.Copy(sourceMem.Array, (long)cc * structureSizeOf, tmpArray, 0L, structureSizeOf);

                            var aIndex = BitConverter.ToUInt32(tmpArray, 0);
                            var bIndex = BitConverter.ToUInt32(tmpArray, 4);
                            var distance = BitConverter.ToSingle(tmpArray, 8);

                            result.Add(
                                new SortItem(aIndex, bIndex, distance));
                        }

                        return result;

                    }
                };

            TestSortAlgorithm(sort);
        }

        private void TestSortAlgorithm(
            Func<List<SortItem>, List<SortItem>> algo)
        {
            Assert.IsNotNull(algo);

            for (var count = 3; count < 67; count++)
            {
                Console.WriteLine("Count = {0}", count);

                var total = (count + 1)*count/2;

                TestOneConfiguration(algo, count, total);
            }
        }

        private static void TestOneConfiguration(
            Func<List<SortItem>, List<SortItem>> algo,
            int count,
            int total)
        {
            //генерируем тестовый набор
            var dataset = new List<SortItem>();
            for (uint a = 0; a < count; a++)
            {
                for (uint b = a; b < count; b++)
                {
                    var i = new SortItem(a, b, a*b/(float) total);
                    dataset.Add(i);
                }
            }

            //перемешиваем
            var rnd = new Random();
            for (int i = 0; i < total; i++)
            {
                if (rnd.NextDouble() >= 0.5d)
                {
                    var newIndex = rnd.Next(dataset.Count);

                    var tmp = dataset[i];
                    dataset[i] = dataset[newIndex];
                    dataset[newIndex] = tmp;
                }
            }

            //выполняем алгоритм
            var result = algo(dataset);

            //проверяем выполнение
            if (result == null)
            {
                Assert.Fail();
            }

            if (result.Count != total)
            {
                Assert.Fail();
            }

            for (var cc = 0; cc < total - 1; cc++)
            {
                var leftKey = ((ulong) result[cc].AIndex << 32) + result[cc].BIndex;
                var rightKey = ((ulong) result[cc + 1].AIndex << 32) + result[cc + 1].BIndex;

                Assert.IsTrue(leftKey < rightKey);
            }

            for (var cc = 0; cc < total; cc++)
            {
                var orig = dataset.Find(j => j.AIndex == result[cc].AIndex && j.BIndex == result[cc].BIndex);
                
                Assert.IsNotNull(orig);

                Assert.AreEqual(orig.Distance, result[cc].Distance);
            }
        }
    }
}
