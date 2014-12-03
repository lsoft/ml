using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.IterateHelper;
using MyNN.Common.Other;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public class AccuracyCalculatorBatchIterator
    {
        public void IterateByBatch(
            IEnumerable<IDataItem> validationData,
            IForwardPropagation forwardPropagation,
            GiveResultDelegate gr
            )
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }
            if (gr == null)
            {
                throw new ArgumentNullException("gr");
            }

            Task task = null;
            try
            {
                const int batchSize = 5000;

                var batchIndex = 0;
                foreach (var validationBatch in validationData.LazySplit(batchSize))
                {
                    var netResults = forwardPropagation.ComputeOutput(validationBatch);

                    if (task != null)
                    {
                        task.Wait();
                        task.Dispose();
                        task = null;
                    }

                    task = new Task(
                        (opair) =>
                        {
                            var pairs = (Pair<List<IDataItem>, List<ILayerState>>)opair;

                            var tValidationItems = pairs.First;
                            var tNetResult = pairs.Second;

                            foreach (var pair in tValidationItems.ZipEqualLength(tNetResult))
                            {
                                var validationItem = pair.Value1;
                                var netResult = pair.Value2;

                                gr(netResult, validationItem);
                            }
                        },
                        new Pair<List<IDataItem>, List<ILayerState>>(validationBatch, netResults)
                        );
                    task.Start();

                    Console.Write("Validated {0}   ", (batchIndex * batchSize));
                    Console.SetCursorPosition(0, Console.CursorTop);

                    batchIndex++;
                }

                if (task != null)
                {
                    task.Wait();
                }
            }
            finally
            {
                if (task != null)
                {
                    task.Dispose();
                }
            }
        }
    }
}