using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using MyNN.Data;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons;

namespace MyNN.NeuralNet.Train.Algo
{
    public abstract class BaseTrainAlgorithm
    {
        protected readonly MultiLayerNeuralNetwork _network;
        protected ILearningAlgorithmConfig _config = null;
        private readonly MultilayerTrainProcessDelegate _validation;
        private readonly bool _enableErrorRecalculate;

        protected BaseTrainAlgorithm(
            MultiLayerNeuralNetwork network, 
            ILearningAlgorithmConfig config, 
            MultilayerTrainProcessDelegate validation,
            bool enableErrorRecalculate = true)
        {
            _network = network;
            _config = config;
            _validation = validation;
            _enableErrorRecalculate = enableErrorRecalculate;
        }

        public void Train(DataSourceDelegate dataSource)
        {
            #region validate

            if (dataSource == null)
            {
                throw new ArgumentNullException("dataSource");
            }

            #endregion

            #region валидируем дефолтовую сеть

            Console.WriteLine("Default net validation results:");

            var preTrainFolder = Path.Combine(_network.WorkFolderPath, "_pretrain");
            Directory.CreateDirectory(preTrainFolder);
            _validation(_network, preTrainFolder, float.MaxValue, false);

            #endregion

            var currentError = float.MaxValue;
            var lastError = 0.0f;
            var epochNumber = 0;

            Console.WriteLine("Predeformation...");

            //запрашиваем данные
            var data = dataSource(epochNumber);

            Console.WriteLine("Start training...");

            if (_config.BatchSize < 1 || _config.BatchSize > data.Count)
            {
                _config.ReassignBatchSize(data.Count);
            }

            //создаем массивы
            this.PreTrainInit(data);

            //цикл по эпохам
            do
            {
                lastError = currentError;

                //скорость обучения на эту эпоху
                var learningRate = _config.LearningRateController.GetLearningRate(epochNumber);
                Console.WriteLine("Epoch learning rate: " + learningRate);

                #region epoche train

                //создаем папку эпохи
                var epocheRoot = Path.Combine(_network.WorkFolderPath, string.Format("epoche {0}", epochNumber));
                Directory.CreateDirectory(epocheRoot);

                //перемешиваем данные для эпохи
                var shuffled = data.CreateShuffledDataSet();

                var dtStart = DateTime.Now;

                //обучаем эпоху
                this.TrainEpoche(shuffled, epocheRoot, learningRate);

                //сколько времени заняла эпоха обучения
                var trainTimeEnd = DateTime.Now;

                #endregion

                #region recalculating error on all data

                if (_enableErrorRecalculate)
                {
                    var realOutput2 = _network.ComputeOutput(data.GetInputPart());

                    //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
                    //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
                    var d = new List<DataItem>(data.Count + 1);
                    for (int i = 0; i < data.Count; i++)
                    {
                        d.Add(new DataItem(data[i].Output, realOutput2[i]));
                    }

                    currentError = d.AsParallel().Sum(
                        j => _config.ErrorFunction.Calculate(j.Input, j.Output));

                    currentError *= 1.0f/data.Count;

                    //regularization term (не оптимизировано, малый эффект)
                    if (Math.Abs(_config.RegularizationFactor) > float.Epsilon)
                    {
                        var reg = _network.Layers.Sum(layer => layer.Neurons.Sum(neuron => neuron.Weights.Sum(weight => weight*weight)));
                        currentError += _config.RegularizationFactor*reg/(2.0f*data.Count);
                    }
                }

                //сколько времени заняла эпоха обучения
                var errorRecalculationTimeEnd = DateTime.Now;

                #endregion

                epochNumber++;

                #region запрашиваем искаженные данные для следующей эпохи

                var deformStart = DateTime.Now;

                data = dataSource(epochNumber);

                //сколько времени заняло искажение данных
                var dtFinish = DateTime.Now;

                #endregion

                #region report epoche results

                Console.WriteLine(
                    "-------------------  "
                    + "Epoch #" + epochNumber.ToString("D7")
                    + "  -----------  "
                    + " Err = " + (currentError == float.MaxValue ? "не вычислено" : currentError.ToString())
                    + "  -------------------");

                Console.WriteLine("Current time: " + DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss"));

                //внешняя функция для обсчета на тестовом множестве
                _validation(_network, epocheRoot, currentError, true);

                var cvFinish = DateTime.Now;

                Console.WriteLine(
                    "   "
                    + "Total: " + (int) ((cvFinish - dtStart).TotalMilliseconds)
                    + "  Train: " + (int) ((trainTimeEnd - dtStart).TotalMilliseconds)
                    + "  ErrRecalc: " + (int) ((errorRecalculationTimeEnd - trainTimeEnd).TotalMilliseconds)
                    + "  Validation: " + (int) ((cvFinish - dtFinish).TotalMilliseconds)
                    + "  Deform: " + (int) ((dtFinish - deformStart).TotalMilliseconds));

                Console.WriteLine(
                    "----------------------------------------------------------------------------------------------");
                Console.WriteLine(string.Empty);


                #endregion

                GC.Collect(0);
                GC.WaitForPendingFinalizers();
                GC.Collect(0);

            } while (epochNumber < _config.MaxEpoches &&
                     currentError > _config.MinError &&
                     Math.Abs(currentError - lastError) > _config.MinErrorChange);
        }

        protected abstract void PreTrainInit(DataSet data);
        protected abstract void TrainEpoche(DataSet data, string epocheRoot, float learningRate);
    }
}
