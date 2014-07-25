using System;
using System.IO;
using System.Linq;
using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation.EpocheTrainer;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.MLP2.Backpropagation
{
    public class BackpropagationAlgorithm : IBackpropagationAlgorithm
    {
        private readonly IRandomizer _randomizer;
        private readonly IMLP _mlp;
        private readonly IValidation _validation;
        private readonly ILearningAlgorithmConfig _config;
        private readonly IBackpropagationEpocheTrainer _backpropagationEpocheTrainer;
        private readonly IArtifactContainer _artifactContainer;

        private IAccuracyRecord _bestAccuracyRecord;

        public BackpropagationAlgorithm(
            IRandomizer randomizer,
            IBackpropagationEpocheTrainer backpropagationEpocheTrainer,
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IValidation validation,
            ILearningAlgorithmConfig config)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (backpropagationEpocheTrainer == null)
            {
                throw new ArgumentNullException("backpropagationEpocheTrainer");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }
            if (validation == null)
            {
                throw new ArgumentNullException("validation");
            }

            _randomizer = randomizer;
            _backpropagationEpocheTrainer = backpropagationEpocheTrainer;
            _artifactContainer = artifactContainer;//.GetChildContainer(mlp.Name);
            _mlp = mlp;
            _validation = validation;
            _config = config;
        }

        public void Train(ITrainDataProvider trainDataProvider)
        {
            if (trainDataProvider == null)
            {
                throw new ArgumentNullException("trainDataProvider");
            }

            ConsoleAmbientContext.Console.WriteLine(
                "BACKPROPAGATION STARTED WITH {0}",
                _mlp.GetLayerInformation());
            ConsoleAmbientContext.Console.WriteLine(
                "EPOCHE TRAINER = {0}, VALIDATION = {1}",
                this._backpropagationEpocheTrainer.GetType().Name,
                this._validation.GetType().Name);

            #region валидируем дефолтовую сеть

            var beforeDefaultValidation = DateTime.Now;

            ConsoleAmbientContext.Console.WriteLine("Default net validation results:");

            var preTrainContainer = _artifactContainer.GetChildContainer("_pretrain");

            _validation.Validate(
                _backpropagationEpocheTrainer.ForwardPropagation,
                null,
                preTrainContainer
                );

            var afterDefaultValidation = DateTime.Now;
            ConsoleAmbientContext.Console.WriteLine("Default net validation takes {0}", (afterDefaultValidation - beforeDefaultValidation));

            #endregion

            var currentError = float.MaxValue;
            var lastError = 0.0f;
            var epochNumber = 0;


            ConsoleAmbientContext.Console.WriteLine("Predeformation...");

            //запрашиваем данные
            var trainData = trainDataProvider.GetDataSet(epochNumber);

            ConsoleAmbientContext.Console.WriteLine("Start training...");

            if (_config.BatchSize < 1 || _config.BatchSize > trainData.Count)
            {
                _config.ReassignBatchSize(trainData.Count);
            }

            //создаем массивы
            this._backpropagationEpocheTrainer.PreTrainInit(trainData);

            //цикл по эпохам
            do
            {
                lastError = currentError;

                ConsoleAmbientContext.Console.WriteLine(
                "---------------------------------------- Epoch #{0} --------------------------------------",
                    epochNumber.ToString("D7"));

                ConsoleAmbientContext.Console.WriteLine("Current time: " + DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss"));

                //скорость обучения на эту эпоху
                var learningRate = _config.LearningRateController.GetLearningRate(epochNumber);
                ConsoleAmbientContext.Console.WriteLine("Epoch learning rate: " + learningRate);

                #region train epoche

                //создаем папку эпохи
                var epocheContainer = _artifactContainer.GetChildContainer(
                    string.Format("epoche {0}", epochNumber));

                //перемешиваем данные для эпохи
                var shuffled = trainData.CreateShuffledDataSet(_randomizer);

                var dtStart = DateTime.Now;

                //обучаем эпоху
                _backpropagationEpocheTrainer.TrainEpoche(shuffled, epocheContainer, learningRate);

                //сколько времени заняла эпоха обучения
                var trainTimeEnd = DateTime.Now;

                #endregion

                #region validation

                //внешняя функция для обсчета на тестовом множестве
                var epocheAccuracyRecord = _validation.Validate(
                    _backpropagationEpocheTrainer.ForwardPropagation,
                    epochNumber,
                    epocheContainer
                    );
                currentError = epocheAccuracyRecord.PerItemError;

                var needToSaveMLP = (_bestAccuracyRecord == null || (epocheAccuracyRecord.IsBetterThan(_bestAccuracyRecord)));

                var cvFinish = DateTime.Now;

                #endregion

                #region correct error with regularization

                //regularization term (не оптимизировано, малый эффект)
                if (Math.Abs(_config.RegularizationFactor) > float.Epsilon)
                {
                    var reg = _mlp.Layers.Sum(layer => layer.Neurons.Sum(neuron => neuron.Weights.Sum(weight => weight*weight)));
                    currentError += _config.RegularizationFactor * reg / (2.0f * trainData.Count);
                }

                //сколько времени заняла эпоха обучения
                var errorRecalculationTimeEnd = DateTime.Now;

                #endregion

                epochNumber++;

                #region запрашиваем искаженные данные для следующей эпохи

                var deformStart = DateTime.Now;

                trainData = trainDataProvider.GetDataSet(epochNumber);

                //сколько времени заняло искажение данных
                var dtFinish = DateTime.Now;

                #endregion

                #region report epoche results

                ConsoleAmbientContext.Console.WriteLine(
                    "                  =========== Per-item error = {0} ===========",
                    (currentError >= float.MaxValue ? "не вычислено" : DoubleConverter.ToExactString(currentError))
                    );

                ConsoleAmbientContext.Console.WriteLine("Current time: " + DateTime.Now.ToString("dd.MM.yyyy HH:mm:ss"));

                ConsoleAmbientContext.Console.WriteLine(
                    "   "
                    + "Total: " + (int) ((dtFinish - dtStart).TotalMilliseconds)
                    + "  Train: " + (int) ((trainTimeEnd - dtStart).TotalMilliseconds)
                    + "  ErrRecalc: " + (int) ((errorRecalculationTimeEnd - cvFinish).TotalMilliseconds)
                    + "  Validation: " + (int) ((cvFinish - trainTimeEnd).TotalMilliseconds)
                    + "  Deform: " + (int) ((dtFinish - deformStart).TotalMilliseconds));

                if (needToSaveMLP)
                {
                    ConsoleAmbientContext.Console.WriteLine("Saved!");
                }

                ConsoleAmbientContext.Console.WriteLine(
                    "----------------------------------------------------------------------------------------------");
                ConsoleAmbientContext.Console.WriteLine(string.Empty);

                #endregion

                #region save mlp if better

                if (needToSaveMLP)
                {
                    _bestAccuracyRecord = epocheAccuracyRecord;
                    
                    epocheContainer.SaveMLP(
                        _backpropagationEpocheTrainer.ForwardPropagation.MLP,
                        epocheAccuracyRecord);
                }

                #endregion

                GC.Collect(0);
                GC.WaitForPendingFinalizers();
                GC.Collect(0);
            } while (epochNumber < _config.MaxEpoches &&
                     currentError > _config.MinError &&
                     Math.Abs(currentError - lastError) > _config.MinErrorChange);
        }

    }
}
