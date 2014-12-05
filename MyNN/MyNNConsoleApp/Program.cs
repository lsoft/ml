using System;
using System.Collections.Generic;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Randomizer;
using MyNNConsoleApp.CTRP;
using MyNNConsoleApp.DBN;
using MyNNConsoleApp.RefactoredForDI;
using OpenCL.Net;

namespace MyNNConsoleApp
{
    class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                //var dil = new CRPDataItemLoader(
                //    "DATA #1",
                //    "__train.bin",
                //    new DataItemFactory()
                //    );

                //var item0 = dil.Load(0);
                //var itemlast = dil.Load(dil.Count - 1);

                //Console.WriteLine(item0);
                //Console.WriteLine(itemlast);

                TrainMLP.DoTrain();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }

        //private static void TestCacheDataIterator(
        //    )
        //{
        //    var listdataitem = new List<IDataItem>();

        //    for (var cc = 0; cc < 15; cc++)
        //    {
        //        var input = new float[] {cc,cc};
        //        var output = new float[] {cc};

        //        listdataitem.Add(
        //            new DataItem(input, output));
        //    }

        //    IDataItemLoader itemLoader = new FromArrayDataItemLoader(
        //        listdataitem,
        //        new DefaultNormalizer()
        //        );

        //    itemLoader = new ShuffleDataItemLoader(
        //        new DefaultRandomizer(123),
        //        itemLoader
        //        );

        //    IDataIterator di = new DataIterator(
        //        itemLoader,
        //        () => new NoConvertDataItemTransformation()
        //        );

        //    di = new CacheDataIterator(100, di);

        //    while(di.MoveNext())
        //    {
        //        Console.WriteLine(
        //            "{0} {1} -> {2}",
        //            di.Current.Input[0],
        //            di.Current.Input[1],
        //            di.Current.Output[0]
        //            );
        //    }

        //    di.Reset();

        //    di.Dispose();
        //}
    }
}
