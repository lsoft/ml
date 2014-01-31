using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Saver;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagation.Validation
{
    public class AutoencoderValidation : IValidation
    {
        private readonly IMLPSaver _mlpSaver;
        private readonly IMetrics _metrics;

        private readonly DataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;

        private float _bestPerItemError = float.MaxValue;
        public float BestPerItemError
        {
            get
            {
                return _bestPerItemError;
            }
        }

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    _validationData.IsAuencoderDataSet;
            }
        }

        public AutoencoderValidation(
            IMLPSaver mlpSaver,
            IMetrics metrics,
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount)
        {
            if (mlpSaver == null)
            {
                throw new ArgumentNullException("mlpSaver");
            }
            if (metrics == null)
            {
                throw new ArgumentNullException("metrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            if (!validationData.IsAuencoderDataSet)
            {
                throw new ArgumentException("Для этого валидатора годятся только датасеты для автоенкодера");
            }

            _mlpSaver = mlpSaver;
            _metrics = metrics;
            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
        }

        public float Validate(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave)
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            #region сохраняем картинки

            var netResults = forwardPropagation.ComputeOutput(_validationData);

            //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
            //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
            var d = new List<DataItem>(_validationData.Count + 1);
            for (int i = 0; i < _validationData.Count; i++)
            {
                d.Add(
                    new DataItem(
                        _validationData[i].Output,
                        netResults[i].State));
            }

            var totalError = d.AsParallel().Sum(
                j => _metrics.Calculate(j.Input, j.Output));

            var perItemError = totalError / _validationData.Count;

            if (_validationData.IsAbleToVisualize)
            {
                if (_validationData.IsAuencoderDataSet)
                {
                    _validationData.SaveAsGrid(
                        Path.Combine(epocheRoot, "grid.bmp"),
                        netResults.ConvertAll(j => j.State).Take(_visualizeAsGridCount).ToList());

                    //со случайного индекса
                    var startIndex = (int)((DateTime.Now.Millisecond/1000f)*(_validationData.Count - _visualizeAsPairCount));

                    var pairList = new List<Pair<float[], float[]>>();
                    for (var cc = startIndex; cc < startIndex + _visualizeAsPairCount; cc++)
                    {
                        var i = new Pair<float[], float[]>(
                            _validationData[cc].Input,
                            netResults[cc].State);
                        pairList.Add(i);
                    }

                    _validationData.SaveAsPairList(
                        Path.Combine(epocheRoot, "reconstruct.bmp"),
                        pairList);
                }
            }

            #endregion

            #region если результат лучше, чем был, то сохраняем его

            if (_bestPerItemError >= perItemError)
            {
                _bestPerItemError = Math.Min(_bestPerItemError, perItemError);

                if (allowToSave)
                {
                    var accuracyRecord = new AccuracyRecord(perItemError);

                    _mlpSaver.Save(
                        epocheRoot,
                        accuracyRecord,
                        forwardPropagation.MLP);

                    ConsoleAmbientContext.Console.WriteLine("Saved!");
                }
            }

            #endregion

            if (!allowToSave)
            {
                ConsoleAmbientContext.Console.WriteLine(
                    "Per item error = {0}",
                    perItemError);
            }

            return perItemError;
        }

    }

}
