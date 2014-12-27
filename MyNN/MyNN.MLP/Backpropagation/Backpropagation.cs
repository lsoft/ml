using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net;

namespace MyNN.MLP.Backpropagation
{
    public class Backpropagation : IBackpropagation
    {
        private readonly IMLP _mlp;
        private readonly IValidation _validation;
        private readonly ILearningAlgorithmConfig _config;
        private readonly IForwardPropagation _inferenceForwardPropagation;
        private readonly IEpocheTrainer _epocheTrainer;
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly IArtifactContainer _artifactContainer;

        private IAccuracyRecord _bestAccuracyRecord;

        public Backpropagation(
            IEpocheTrainer epocheTrainer,
            IMLPContainerHelper mlpContainerHelper,
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IValidation validation,
            ILearningAlgorithmConfig config,
            IForwardPropagation inferenceForwardPropagation
            )
        {
            if (epocheTrainer == null)
            {
                throw new ArgumentNullException("epocheTrainer");
            }
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }
            if (validation == null)
            {
                throw new ArgumentNullException("validation");
            }
            if (inferenceForwardPropagation == null)
            {
                throw new ArgumentNullException("inferenceForwardPropagation");
            }

            _epocheTrainer = epocheTrainer;
            _mlpContainerHelper = mlpContainerHelper;
            _artifactContainer = artifactContainer;
            _mlp = mlp;
            _validation = validation;
            _config = config;
            _inferenceForwardPropagation = inferenceForwardPropagation;
        }

        public IAccuracyRecord Train(IDataSetProvider dataSetProvider)
        {
            if (dataSetProvider == null)
            {
                throw new ArgumentNullException("dataSetProvider");
            }

            ConsoleAmbientContext.Console.WriteLine(
                "BACKPROPAGATION STARTED WITH {0}",
                _mlp.GetLayerInformation());

            #region валидируем дефолтовую сеть

            var beforeDefaultValidation = DateTime.Now;

            ConsoleAmbientContext.Console.WriteLine("Default net validation results:");

            var preTrainContainer = _artifactContainer.GetChildContainer("_pretrain");

            _validation.Validate(
                _inferenceForwardPropagation,
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

            //запрашиваем данные (уже перемешанные)
            var trainData = dataSetProvider.GetDataSet(epochNumber);

            if (trainData.OutputLength != _mlp.Layers.Last().NonBiasNeuronCount)
            {
                throw new Exception(
                    string.Format(
                        "На последнем слое сети {1} нейронов, а у датасета OutputLength = {0}",
                        trainData.OutputLength,
                        _mlp.Layers.Last().NonBiasNeuronCount));
            }


            ConsoleAmbientContext.Console.WriteLine("Start training...");

            if (_config.BatchSize < 1 || _config.BatchSize > trainData.Count)
            {
                _config.ReassignBatchSize(trainData.Count);
            }

            IAccuracyRecord epocheAccuracyRecord = null;

            //создаем массивы
            this._epocheTrainer.PreTrainInit(trainData);

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

                var dtStart = DateTime.Now;

                //обучаем эпоху
                _epocheTrainer.TrainEpoche(trainData, epocheContainer, learningRate);

                //сколько времени заняла эпоха обучения
                var trainTimeEnd = DateTime.Now;

                #endregion

                #region validation

                //внешняя функция для обсчета на тестовом множестве
                epocheAccuracyRecord = _validation.Validate(
                    _inferenceForwardPropagation,
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

                //получаем данные для следующей эпохи (уже перемешанные)
                trainData = dataSetProvider.GetDataSet(epochNumber);

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
                    + "Total: " + (int)((dtFinish - dtStart).TotalMilliseconds)
                    + "  Train: " + (int)((trainTimeEnd - dtStart).TotalMilliseconds)
                    + "  ErrRecalc: " + (int)((errorRecalculationTimeEnd - cvFinish).TotalMilliseconds)
                    + "  Validation: " + (int)((cvFinish - trainTimeEnd).TotalMilliseconds)
                    + "  Deform: " + (int)((dtFinish - deformStart).TotalMilliseconds));

                if (needToSaveMLP)
                {
                    ConsoleAmbientContext.Console.WriteWarning("Saved!");
                }

                ConsoleAmbientContext.Console.WriteLine(
                    "----------------------------------------------------------------------------------------------");
                ConsoleAmbientContext.Console.WriteLine(string.Empty);

                #endregion

                #region save mlp if better

                if (needToSaveMLP)
                {
                    _bestAccuracyRecord = epocheAccuracyRecord;
                    
                    _mlpContainerHelper.SaveMLP(
                        epocheContainer,
                        _inferenceForwardPropagation.MLP,
                        epocheAccuracyRecord);
                }

                #endregion

                GC.Collect(0);
                GC.WaitForPendingFinalizers();
                GC.Collect(0);
            } while (epochNumber < _config.MaxEpoches &&
                     currentError > _config.MinError &&
                     Math.Abs(currentError - lastError) > _config.MinErrorChange);

            return
                epocheAccuracyRecord;
        }

    }
}
