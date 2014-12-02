using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;

namespace MyNN.Boosting.SAMME
{
    public class SAMME
    {
        /// <summary>
        /// Количество данных для обучения
        /// </summary>
        private int _n;

        /// <summary>
        /// Длина входного вектора
        /// </summary>
        private int _i;

        /// <summary>
        /// Длина выходного вектора
        /// </summary>
        private int _o;

        private readonly IEpocheDataProvider _epocheDataProvider;
        private readonly IEpocheTrainer _epocheTrainer;

        private readonly SAMMEClassifierSet _currentClassifierSet;

        private SAMMEClassifierSet _bestClassifierSet;

        public SAMME(
            IEpocheDataProvider epocheDataProvider,
            IEpocheTrainer epocheTrainer)
        {
            if (epocheDataProvider == null)
            {
                throw new ArgumentNullException("epocheDataProvider");
            }
            if (epocheTrainer == null)
            {
                throw new ArgumentNullException("epocheTrainer");
            }

            _epocheDataProvider = epocheDataProvider;
            _epocheTrainer = epocheTrainer;

            _currentClassifierSet = new SAMMEClassifierSet();
            _bestClassifierSet = _currentClassifierSet;
        }

        public SAMMEClassifierSet Train(
            IDataSet trainData,
            IDataSet validationData, 
            int epocheThreshold,
            float errorThreshold)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _n = trainData.Count;
            _i = trainData.InputLength;
            _o = trainData.OutputLength;

            var n1 = 1f/_n;

            Console.WriteLine("Start SAMME boosting...");

            //Готовим веса
            var w = new float[_n];
            for (var cc = 0; cc < w.Length; cc++)
            {
                w[cc] = n1;
            }

            //готовим датасет
            var inputs = trainData.Select(j => j.Input.ToList().ConvertAll(k => (double) k).ToArray()).ToArray();
            var labels = trainData.Select(j => j.OutputIndex).ToArray();

            var bestCorrectCount = 0;

            //цикл по эпохам
            for (var m = 0; m < epocheThreshold; m++)
            {
                Console.WriteLine("Epoche " + m);

                //формируем набор для обучения
                List<double[]> epocheInputs = null;
                List<int> epocheLabels = null;

                //формируем датасет для эпохи
                _epocheDataProvider.GetEpocheDataSet(
                    m,
                    inputs,
                    labels,
                    out epocheInputs,
                    out epocheLabels,
                    w);

                //обучаем классификатор
                var classifier = _epocheTrainer.TrainEpocheClassifier(
                    epocheInputs, 
                    epocheLabels, 
                    this._o,
                    this._i);

                //считаем ошибку
                var sumW0 = w.Sum();
                var err = 0f;
                for (var cc = 0; cc < _n; cc++)
                {
                    err += (classifier.Compute(inputs[cc]) != labels[cc] ? w[cc] : 0f);
                }
                err /= sumW0;

                Console.WriteLine("SAMME error: " + err);

                if (err < errorThreshold)
                {
                    break;
                }

                //считаем альфу
                var alpha = (float)(Math.Log((1 - err)/err) + Math.Log(_o - 1));

                //обновляем веса
                for (var cc = 0; cc < _n; cc++)
                {
                    w[cc] *= (float)Math.Exp(classifier.Compute(inputs[cc]) != labels[cc] ? alpha : 0f);
                }

                //нормализуем w
                var sumW1 = w.Sum();
                for (var cc = 0; cc < w.Length; cc++)
                {
                    w[cc] /= sumW1;
                }

                //считаем 
                _currentClassifierSet.Add(classifier, alpha);

                var currentCorrectCount = Validate(this._currentClassifierSet, validationData);
                if (currentCorrectCount >= bestCorrectCount)
                {
                    //лучше!
                    bestCorrectCount = currentCorrectCount;
                    _bestClassifierSet = new SAMMEClassifierSet(_currentClassifierSet);
                }

                Console.WriteLine(string.Empty);
            }

            //валидация на лучшем
            Validate(_bestClassifierSet, validationData);

            Console.WriteLine("SAMME boosting finished");

            return _bestClassifierSet;
        }

        private int Validate(
            SAMMEClassifierSet classifierSet,
            IEnumerable<IDataItem> validationDataList
            )
        {
            if (classifierSet == null)
            {
                throw new ArgumentNullException("classifierSet");
            }
            if (validationDataList == null)
            {
                throw new ArgumentNullException("validationDataList");
            }

            var success = 0;
            var total = 0;

            foreach (var v in validationDataList)
            {
                var ri = classifierSet.Classify(v.Input, this._o);

                if (ri == v.OutputIndex)
                {
                    success++;
                }

                total++;
            }

            Console.WriteLine("Success {0} of {1}", success, total);

            return success;
        }

    }
}
