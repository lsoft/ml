using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM
{
    public class RestrictedBoltzmannMachine : IRestrictedBoltzmannMachine
    {
        public IRandomizer Randomizer
        {
            get;
            private set;
        }

        public int VisibleNeuronCount
        {
            get;
            private set;
        }

        public int HiddenNeuronCount
        {
            get;
            private set;
        }

        public CLProvider CLProvider
        {
            get;
            private set;
        }

        private readonly OpenCL.Net.Wrapper.Mem.Mem<float> _input;
        private readonly OpenCL.Net.Wrapper.Mem.Mem<float> _nabla;

        public OpenCL.Net.Wrapper.Mem.Mem<float> Visible
        {
            get;
            private set;
        }

        public OpenCL.Net.Wrapper.Mem.Mem<float> Hidden0
        {
            get;
            private set;
        }

        public OpenCL.Net.Wrapper.Mem.Mem<float> Hidden1
        {
            get;
            private set;
        }
        
        public OpenCL.Net.Wrapper.Mem.Mem<float> Weights
        {
            get;
            private set;
        }
        
        public OpenCL.Net.Wrapper.Mem.Mem<float> Randoms
        {
            get;
            private set;
        }

        private OpenCL.Net.Wrapper.Mem.Mem<float> _inputBufferForVisibleFreeEnergy;
        private OpenCL.Net.Wrapper.Mem.Mem<float> _resultBufferForVisibleFreeEnergy;

        private const int SizeOfResultBufferForVisibleFreeEnergy = 500;

        private readonly Kernel _clearNabla, _clearNabla4;
        private readonly Kernel _errorCompute;
        private readonly Kernel _changeWeight, _changeWeight4;

        private readonly Kernel _calculateVisibleFreeEnergy;

        public Kernel ComputeVisible
        {
            get;
            private set;
        }
        
        public Kernel ComputeHidden
        {
            get;
            private set;
        }
        
        public Kernel SampleHidden
        {
            get;
            private set;
        }
        
        public Kernel SampleVisible
        {
            get;
            private set;
        }

        public int RandomCount
        {
            get;
            private set;
        }

        private IRBMNegativeSampler _sampler;

        public RestrictedBoltzmannMachine(
            IRandomizer randomizer,
            CLProvider clProvider,
            int visibleNeuronCount,
            int hiddenNeuronCount)
        {

            #region validate

            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            if (visibleNeuronCount <= 0 || hiddenNeuronCount <= 0)
            {
                throw new ArgumentException("visibleNeuronCount <= 0 || hiddenNeuronCount <= 0");
            }

            #endregion

            Randomizer = randomizer;

            this.RandomCount = 16384;

            VisibleNeuronCount = visibleNeuronCount + 1; //bias neuron
            HiddenNeuronCount = hiddenNeuronCount + 1; //bias neuron

            CLProvider = clProvider;

            //создаем массивы данных
            _input = CLProvider.CreateFloatMem(VisibleNeuronCount, MemFlags.CopyHostPtr | MemFlags.ReadOnly);
            Visible = CLProvider.CreateFloatMem(VisibleNeuronCount, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            Hidden0 = CLProvider.CreateFloatMem(HiddenNeuronCount, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            Hidden1 = CLProvider.CreateFloatMem(HiddenNeuronCount, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            Weights = CLProvider.CreateFloatMem(VisibleNeuronCount * HiddenNeuronCount, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            Randoms = CLProvider.CreateFloatMem(RandomCount, MemFlags.CopyHostPtr | MemFlags.ReadOnly);
            _nabla = CLProvider.CreateFloatMem(VisibleNeuronCount * HiddenNeuronCount, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _inputBufferForVisibleFreeEnergy = CLProvider.CreateFloatMem(VisibleNeuronCount * SizeOfResultBufferForVisibleFreeEnergy, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            _resultBufferForVisibleFreeEnergy = CLProvider.CreateFloatMem(SizeOfResultBufferForVisibleFreeEnergy, MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            //модифицируем тексты кернелов
            this._kernelsSource = this._kernelsSource.Replace(
                "{0}",
                VisibleNeuronCount.ToString());

            //создаем кернелы
            _clearNabla = CLProvider.CreateKernel(_kernelsSource, "ClearKernel");
            _clearNabla4 = CLProvider.CreateKernel(_kernelsSource, "ClearKernel4");
            ComputeVisible = CLProvider.CreateKernel(_kernelsSource, "ComputeVisible");
            ComputeHidden = CLProvider.CreateKernel(_kernelsSource, "ComputeHidden");
            SampleHidden = CLProvider.CreateKernel(_kernelsSource, "SampleHidden");
            SampleVisible = CLProvider.CreateKernel(_kernelsSource, "SampleVisible");
            _errorCompute = CLProvider.CreateKernel(_kernelsSource, "ErrorCompute");
            _changeWeight = CLProvider.CreateKernel(_kernelsSource, "ChangeWeight");
            _changeWeight4 = CLProvider.CreateKernel(_kernelsSource, "ChangeWeight4");
            _calculateVisibleFreeEnergy = CLProvider.CreateKernel(_kernelsSource, "CalculateVisibleFreeEnergy");

            //инициализируем веса
            for (var cc = 0; cc < Weights.Array.Length; cc++)
            {
                Weights.Array[cc] = Randomizer.Next() * 0.02f - 0.01f;
            }

            //заполняем bias входных объектов
            _input.Array[VisibleNeuronCount - 1] = 1f;
            Visible.Array[VisibleNeuronCount - 1] = 1f;

            //заполняем bias выходных объектов
            Hidden0.Array[HiddenNeuronCount - 1] = 1f;
            Hidden1.Array[HiddenNeuronCount - 1] = 1f;

            //заполняем рандом
            for (var cc = 0; cc < Randoms.Array.Length; cc++)
            {
                Randoms.Array[cc] = Randomizer.Next();
            }

            //отправляем видимое (пустой массив + биас)
            Visible.Write(BlockModeEnum.Blocking);

            //отправляем скрытые (пустой массив + биас)
            Hidden0.Write(BlockModeEnum.Blocking);
            Hidden1.Write(BlockModeEnum.Blocking);

            //Отправляем веса
            Weights.Write(BlockModeEnum.Blocking);

            //Отправляем рандом
            Randoms.Write(BlockModeEnum.Blocking);

        }

        public void SetNegativeSampler(IRBMNegativeSampler sampler)
        {
            #region validate

            if (sampler == null)
            {
                throw new ArgumentNullException("sampler");
            }

            #endregion

            _sampler = sampler;
        }

        public void Train(
            DataSet trainData,
            DataSet validationData,
            int batchSize,
            ILearningRate learningRateController,
            float errorThreshold,
            int epochThreshold,
            string artifactFolderRoot,
            IFeatureExtractor featureExtractor,
            IImageReconstructor imageReconstructor,
            int reconstructedImageCount,
            int maxGibbsChainLength)
        {
            #region validate

            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }
            if (featureExtractor == null)
            {
                throw new ArgumentNullException("featureExtractor");
            }
            if (imageReconstructor == null)
            {
                throw new ArgumentNullException("imageReconstructor");
            }

            #endregion

            Console.WriteLine(this._sampler.Name + " starts");

            Directory.CreateDirectory(artifactFolderRoot);

            #region формируем наборы для вычисления свободной энергии

            DataSet trainFreeEnergySet = null;
            DataSet validationFreeEnergySet = null;

            if (trainData.Count > validationData.Count)
            {
                trainFreeEnergySet = new DataSet(trainData, validationData.Count);
                validationFreeEnergySet = validationData;
            }
            else
            {
                trainFreeEnergySet = trainData;
                validationFreeEnergySet = new DataSet(validationData, trainData.Count);
            }

            #endregion

            #region free energy

            CalculateFreeEnergy(
                artifactFolderRoot,
                trainFreeEnergySet,
                validationFreeEnergySet);

            #endregion

            #region sampler prepare train

            this._sampler.PrepareTrain(batchSize);

            #endregion

            CLProvider.QueueFinish();

            #region сохраняем веса для построения гистограммы

            this.SaveWeightHistogram(
                Path.Combine(artifactFolderRoot, "__start_weights_hist.csv"));

            #endregion

            var lastErrors = new List<float>();
            var epochNumber = 0;
            while (true)
            {
                #region подготовка к эпохе

                //стартовая ошибка эпохи = 0
                var epocheError = 0.0f;

                //создаем папку артефактов эпохи
                var epocheRoot = Path.Combine(artifactFolderRoot, "epoche " + epochNumber);
                if (Directory.Exists(epocheRoot))
                {
                    Directory.Delete(epocheRoot, true);
                }
                Directory.CreateDirectory(epocheRoot);

                //выводим диагностические данные эпохи
                Console.WriteLine("Epoch " + epochNumber);

                //скорость обучения на эту эпоху
                var learningRate = learningRateController.GetLearningRate(epochNumber);
                Console.WriteLine("Epoch learning rate: " + learningRate);

                #endregion

                //обучаем
                var enumerator = trainData.CreateShuffledDataSet(Randomizer).GetEnumerator();

                var begin = DateTime.Now;

                var indexInBatch = 0;
                var continueProcess = true;
                while (continueProcess)
                {
                    #region очищаем наблу

                    if (CLProvider.ChoosedDeviceType == DeviceType.Cpu)
                    {
                        const int perClearKernelFloats = 1500; //(должно быть кратно 4м!!!)

                        var kernelCount = VisibleNeuronCount * HiddenNeuronCount / perClearKernelFloats;
                        if ((VisibleNeuronCount * HiddenNeuronCount) % perClearKernelFloats > 0)
                        {
                            kernelCount++;
                        }

                        _clearNabla4
                            .SetKernelArgMem(0, _nabla)
                            .SetKernelArg(1, 4, VisibleNeuronCount * HiddenNeuronCount)
                            .SetKernelArg(2, 4, perClearKernelFloats)
                            .EnqueueNDRangeKernel(kernelCount);
                    }
                    else
                    {
                        _clearNabla
                            .SetKernelArgMem(0, _nabla)
                            .EnqueueNDRangeKernel(VisibleNeuronCount * HiddenNeuronCount);
                    }

                    #endregion

                    #region sampler prepare batch

                    this._sampler.PrepareBatch();

                    #endregion

                    //цикл внутри батча
                    for (var batchIndex = 0; batchIndex < batchSize; batchIndex++)
                    {
                        if (!enumerator.MoveNext())
                        {
                            continueProcess = false;
                            break;
                        }

                        indexInBatch++;

                        var trainItem = enumerator.Current;

                        //gibbs sampling

                        //заполняем видимое
                        Array.Copy(trainItem.Input, _input.Array, VisibleNeuronCount - 1);
                        _input.Write(BlockModeEnum.Blocking);

                        var randomIndex = Randomizer.Next(RandomCount);

                        SampleHidden
                            .SetKernelArgMem(0, Hidden0)
                            .SetKernelArgMem(1, _input)

                            .SetKernelArgMem(2, Weights)
                            .SetKernelArgMem(3, Randoms)

                            .SetKernelArg(4, 4, HiddenNeuronCount)
                            .SetKernelArg(5, 4, VisibleNeuronCount)

                            .SetKernelArg(6, 4, randomIndex)
                            .SetKernelArg(7, 4, RandomCount)

                            .EnqueueNDRangeKernel(HiddenNeuronCount - 1); //without bias

                        //get negative sample
                        this._sampler.GetNegativeSample(
                            batchIndex,
                            maxGibbsChainLength);

                        //считаем разницу и записываем ее в наблу
                        _errorCompute
                            .SetKernelArgMem(0, Hidden0)
                            .SetKernelArgMem(1, _input)

                            .SetKernelArgMem(2, Hidden1)
                            .SetKernelArgMem(3, Visible)

                            .SetKernelArgMem(4, _nabla)

                            .SetKernelArg(5, 4, HiddenNeuronCount)
                            .SetKernelArg(6, 4, VisibleNeuronCount)

                            .EnqueueNDRangeKernel(HiddenNeuronCount - 1); //without bias

                        CLProvider.QueueFinish();

                        #region hidden neuron activities logging

                        if (indexInBatch == 1)
                        {
                            Hidden1.Read(BlockModeEnum.Blocking);

                            File.AppendAllText(
                                Path.Combine(artifactFolderRoot, "_hidden1.csv"),
                                string.Join(
                                    ";",
                                    Hidden1.Array.ToList().ConvertAll(k => k.ToString())) + "\r\n");
                        }

                        #endregion

                    }

                    _sampler.BatchFinished();

                    if (CLProvider.ChoosedDeviceType == DeviceType.Cpu)
                    {
                        const int perUpdateKernelFloats = 1500;//(должно быть кратно 4м!!!)

                        var updateKernelCount = VisibleNeuronCount * HiddenNeuronCount / perUpdateKernelFloats;
                        if ((VisibleNeuronCount * HiddenNeuronCount) % perUpdateKernelFloats > 0)
                        {
                            updateKernelCount++;
                        }

                        _changeWeight4
                            .SetKernelArgMem(0, Weights)
                            .SetKernelArgMem(1, _nabla)

                            .SetKernelArg(2, 4, VisibleNeuronCount * HiddenNeuronCount)
                            .SetKernelArg(3, 4, perUpdateKernelFloats)
                            .SetKernelArg(4, 4, learningRate)

                            .EnqueueNDRangeKernel(updateKernelCount);
                    }
                    else
                    {
                        _changeWeight
                            .SetKernelArgMem(0, Weights)
                            .SetKernelArgMem(1, _nabla)

                            .SetKernelArg(2, 4, learningRate)

                            .EnqueueNDRangeKernel(VisibleNeuronCount * HiddenNeuronCount);
                    }
                }

                //конец эпохи
                CLProvider.QueueFinish();

                var afterTrain = DateTime.Now;
                var trainDiff = (afterTrain - begin);

                #region validation error calculate

                var indexof = 0;
                foreach (var d in validationData)
                {
                    //заполняем видимое
                    Array.Copy(d.Input, Visible.Array, VisibleNeuronCount - 1);
                    Visible.Write(BlockModeEnum.Blocking);

                    ComputeHidden
                        .SetKernelArgMem(0, Hidden0)
                        .SetKernelArgMem(1, Visible)

                        .SetKernelArgMem(2, Weights)

                        .SetKernelArg(3, 4, HiddenNeuronCount)
                        .SetKernelArg(4, 4, VisibleNeuronCount)

                        .EnqueueNDRangeKernel(HiddenNeuronCount - 1); //without bias

                    ComputeVisible
                        .SetKernelArgMem(0, Hidden0)
                        .SetKernelArgMem(1, Visible)

                        .SetKernelArgMem(2, Weights)

                        .SetKernelArg(3, 4, HiddenNeuronCount)
                        .SetKernelArg(4, 4, VisibleNeuronCount)

                        .EnqueueNDRangeKernel(VisibleNeuronCount - 1); //without bias

                    CLProvider.QueueFinish();

                    Visible.Read(BlockModeEnum.Blocking);

                    var sqdiff = 0.0f;
                    for (var cc = 0; cc < Visible.Array.Length - 1; cc++)
                    {
                        var dln = (Visible.Array[cc] - d.Input[cc]);
                        sqdiff += dln * dln;
                    }
                    epocheError += (float)Math.Sqrt(sqdiff);


                    #region reconstruct

                    if (indexof < reconstructedImageCount)
                    {
                        imageReconstructor.AddPair(
                            indexof,
                            Visible.Array);
                    }
                    indexof++;

                    #endregion
                }
                imageReconstructor.GetReconstructedBitmap().Save(epocheRoot + "/reconstruct.bmp");

                #endregion

                var afterErrorCalculate = DateTime.Now;
                var errorCalculateDiff = (afterErrorCalculate - afterTrain);

                #region free energy

                CalculateFreeEnergy(
                    artifactFolderRoot,
                    trainFreeEnergySet,
                    validationFreeEnergySet);

                #endregion

                #region features extract

                var h = new float[Hidden0.Array.Length];
                for (var cc = 0; cc < HiddenNeuronCount - 1; cc++)
                {
                    h[cc] = 1;

                    var v = ComputeVisibleFromHidden(h);

                    featureExtractor.AddFeature(v);

                    h[cc] = 0;
                }
                featureExtractor.GetFeatureBitmap().Save(epocheRoot + "/" + "feature.bmp");

                #endregion

                File.AppendAllLines(
                    Path.Combine(artifactFolderRoot, "_err.csv"),
                    new string[] { epocheError.ToString() });

                var till = DateTime.Now;
                var featureDiff = (till - afterErrorCalculate);
                var fullDiff = (till - begin);

                Console.WriteLine(
                    string.Format(
                        "Train timeout: {0}    Error timeout: {1}    Feature extract timeout: {2}    Total timeout {3}    Error: {4}",
                        trainDiff,
                        errorCalculateDiff,
                        featureDiff,
                        fullDiff,
                        epocheError));

                var epocheResult = string.Format(@"
Epoche {0},
    training starts {1} 
    training finished {2}, 
    training timeout: {3}
    error timeout: {4}
    feature extract timeout: {5}
    total timeout: {6}
Error: {7}
",
                    epochNumber,
                    begin,
                    afterTrain,
                    trainDiff,
                    errorCalculateDiff,
                    featureDiff,
                    fullDiff,
                    epocheError);
                File.WriteAllText(epocheRoot + "/result.txt", epocheResult);

                Weights.Read(BlockModeEnum.Blocking);

                #region сохраняем веса в бинарном виде

                SaveWeights(epocheRoot + "/" + "weights.bin");

                #endregion

                #region сохраняем веса для построения гистограммы

                this.SaveWeightHistogram(
                    epocheRoot + "/" + "_weights_hist.csv");

                #endregion

                #region считаем ошибки и определяем, надо ли завершаться

                //считаем ошибки
                if (lastErrors.Count > 20)
                {
                    lastErrors.RemoveRange(0, lastErrors.Count - 20);
                }
                lastErrors.Add(epocheError);

                //определяем, что ошибка за последние 10 раундов упала меньше, чем errorThreshold за предыдущие 10
                if (lastErrors.Count >= 20)
                {
                    var avg0 = lastErrors.Take(10).Average();
                    var avg1 = lastErrors.Skip(10).Take(10).Average();

                    if (avg1 > 0)
                    {
                        var d = Math.Abs((avg0 - avg1) / avg0);
                        if (d < errorThreshold)
                        {
                            Console.WriteLine("Training finished by errorThreshold.");
                            break;
                        }
                    }
                }

                //ограничение по количеству эпох
                if (epochNumber >= epochThreshold)
                {
                    Console.WriteLine("Training finished by epochThreshold.");
                    break;
                }

                #endregion

                epochNumber++;

                Console.WriteLine(string.Empty);
            }

            this.Weights.Read(BlockModeEnum.Blocking);

            #region сохраняем веса для построения гистограммы

            this.SaveWeightHistogram(
                Path.Combine(artifactFolderRoot, "__finish_weights_hist.csv"));

            #endregion
        }

        public void LoadWeights(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            var w = SerializationHelper.LoadFromFile<float[]>(filename);

            if (w.Length != Weights.Array.Length)
            {
                throw new InvalidOperationException("несовпадение количества весов");
            }

            Array.Copy(w, Weights.Array, w.Length);

            //Отправляем веса
            Weights.Write(BlockModeEnum.Blocking);

            CLProvider.QueueFinish();
        }

        public void SaveWeights(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            new SerializationHelper().SaveToFile(this.Weights.Array, filename);
        }

        public void SaveWeightHistogram(string filename)
        {
            #region validate

            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }

            #endregion

            var discreteWeights =
                from w in this.Weights.Array
                select ((int)(w * 20)) / 20f;

            var countw =
                from w in discreteWeights
                group w by w
                    into wgroup
                    orderby wgroup.Key ascending
                    select wgroup;

            //вставляем строку (с помощью нее легче в екселе построить правильную гистограмму)
            File.AppendAllText(
                filename,
                ";Weight Count\r\n");

            File.AppendAllLines(
                filename,
                countw.ToList().ConvertAll(j => j.Key.ToString() + ";" + j.Count().ToString()));
        }
        
        public void CalculateFreeEnergy(
            string artifactFolderRoot,
            DataSet trainFreeEnergySet,
            DataSet validationFreeEnergySet)
        {
            #region validate

            if (artifactFolderRoot == null)
            {
                throw new ArgumentNullException("artifactFolderRoot");
            }

            if (trainFreeEnergySet == null)
            {
                throw new ArgumentNullException("trainFreeEnergySet");
            }

            if (validationFreeEnergySet == null)
            {
                throw new ArgumentNullException("validationFreeEnergySet");
            }

            #endregion

            var feTrain = CalculateFreeEnergySet(this.Weights, trainFreeEnergySet);
            var feTrainPerItem = feTrain / trainFreeEnergySet.Count;
            Console.WriteLine(
                "TRAIN: total free energy {0}, per-item free energy {1}",
                feTrain,
                feTrainPerItem);

            var feValidation = CalculateFreeEnergySet(this.Weights, validationFreeEnergySet);
            var feValidationPerItem = feValidation / validationFreeEnergySet.Count;
            Console.WriteLine(
                "VALIDATION: total free energy {0}, per-item free energy {1}",
                feValidation,
                feValidationPerItem);

            var diffPerItem = Math.Abs(feTrainPerItem - feValidationPerItem);
            Console.WriteLine(
                "Per-item diff free energy {0}",
                diffPerItem);

            File.AppendAllLines(
                Path.Combine(artifactFolderRoot, "_free_energy.csv"),
                new string[]
                {
                    feTrain.ToString() + ";" +
                    feValidation.ToString() + ";" +
                    feTrainPerItem.ToString()  + ";" +
                    feValidationPerItem.ToString() + ";" +
                    diffPerItem.ToString()
                });
        }

        public float CalculateFreeEnergySet(
            OpenCL.Net.Wrapper.Mem.Mem<float> weights,
            DataSet data)
        {
            #region validate

            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }

            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            #endregion

            var sumFreeEnergy = 0f;

            var enumerator = data.GetEnumerator();
            var continueSign = true;
            while (continueSign)
            {
                var factualBufferSize = 0;
                for (var bi = 0; bi < Math.Min(data.Count, SizeOfResultBufferForVisibleFreeEnergy); bi++, factualBufferSize++)
                {
                    if (enumerator.MoveNext())
                    {
                        var c = enumerator.Current;

                        c.Input.CopyTo(
                            _inputBufferForVisibleFreeEnergy.Array,
                            this.VisibleNeuronCount * bi);

                        _inputBufferForVisibleFreeEnergy.Array[this.VisibleNeuronCount * (bi + 1) - 1] = 1f;//bias
                    }
                    else
                    {
                        continueSign = false;
                        break;
                    }
                }

                //буфер заполнили
                if (factualBufferSize > 0)
                {
                    _inputBufferForVisibleFreeEnergy.Write(BlockModeEnum.Blocking);

                    //выполняем
                    _calculateVisibleFreeEnergy
                        .SetKernelArgMem(0, _inputBufferForVisibleFreeEnergy)
                        .SetKernelArgMem(1, weights)
                        .SetKernelArgMem(2, _resultBufferForVisibleFreeEnergy)

                        .SetKernelArg(3, 4, HiddenNeuronCount)
                        .SetKernelArg(4, 4, VisibleNeuronCount)

                        .EnqueueNDRangeKernel(factualBufferSize);

                    CLProvider.QueueFinish();

                    _resultBufferForVisibleFreeEnergy.Read(BlockModeEnum.Blocking);

                    for (var cc = 0; cc < factualBufferSize; cc++)
                    {
                        sumFreeEnergy += _resultBufferForVisibleFreeEnergy.Array[cc];
                    }
                }
            }

            return sumFreeEnergy;
        }

        ////это c# копия алгоритма, сделанного на opencl, проверял - результат вроде дают одинаковый
        //public float CalculateFreeEnergySet(
        //    OpenCL.Net.Wrapper.Mem.Mem<float> weights,
        //    DataSet data)
        //{
        //    var sumFreeEnergy = 0f;
        //    foreach (var vd in data)
        //    {
        //        var vis = 0f;
        //        for (var i = 0; i < VisibleNeuronCount - 1; i++)
        //        {
        //            var vi = vd.Input[i];
        //            var ai = weights.Array[weights.Array.Length - VisibleNeuronCount + i];

        //            var visPart = vi * ai;
        //            vis += visPart;
        //        }

        //        var hid = 0f;
        //        for (var j = 0; j < HiddenNeuronCount - 1; j++)
        //        {
        //            var xj = 0f;
        //            for (var i = 0; i < VisibleNeuronCount; i++)
        //            {
        //                var vi =
        //                    i < (VisibleNeuronCount - 1)
        //                        ? vd.Input[i]
        //                        : 1f;
        //                var wij = weights.Array[
        //                    //HiddenNeuronCount * i + j];
        //                    VisibleNeuronCount * j + i];

        //                xj += vi * wij;
        //            }

        //            var expxj = Math.Exp(xj);
        //            var hidPart = Math.Log(1 + expxj);

        //            hid += (float)hidPart;
        //        }

        //        var freeEnergy = -vis - hid;
        //        sumFreeEnergy += freeEnergy;
        //    }

        //    return sumFreeEnergy;
        //}

        public DataSet ExecuteSampleHidden(
            DataSet trainData)
        {
            #region validate

            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }

            #endregion

            var preResult = new List<DataItem>();

            foreach (var data in trainData)
            {
                var randomIndex = Randomizer.Next(RandomCount);

                Array.Copy(data.Input, Visible.Array, data.Input.Length);
                Visible.Write(BlockModeEnum.Blocking);

                SampleHidden
                    .SetKernelArgMem(0, Hidden1)
                    .SetKernelArgMem(1, Visible)

                    .SetKernelArgMem(2, Weights)
                    .SetKernelArgMem(3, Randoms)

                    .SetKernelArg(4, 4, HiddenNeuronCount)
                    .SetKernelArg(5, 4, VisibleNeuronCount)

                    .SetKernelArg(6, 4, randomIndex)
                    .SetKernelArg(7, 4, RandomCount)

                    .EnqueueNDRangeKernel(HiddenNeuronCount - 1); //without bias

                CLProvider.QueueFinish();

                Hidden1.Read(BlockModeEnum.Blocking);

                var a = new float[Hidden1.Array.Length];
                Array.Copy(Hidden1.Array, a, Hidden1.Array.Length);

                var r = new DataItem(a, data.Output);
                preResult.Add(r);
            }

            return
                new DataSet(preResult);
        }

        public float[] ComputeVisibleFromHidden(float[] d)
        {
            #region validate

            if (d == null)
            {
                throw new ArgumentNullException("d");
            }

            #endregion

            Array.Copy(d, Hidden0.Array, d.Length);
            Hidden0.Write(BlockModeEnum.Blocking);

            ComputeVisible
                .SetKernelArgMem(0, Hidden0)
                .SetKernelArgMem(1, Visible)
                
                .SetKernelArgMem(2, Weights)
                
                .SetKernelArg(3, 4, HiddenNeuronCount)
                .SetKernelArg(4, 4, VisibleNeuronCount)
                
                .EnqueueNDRangeKernel(VisibleNeuronCount - 1); //without bias

            CLProvider.QueueFinish();

            Visible.Read(BlockModeEnum.Blocking);

            return
                Visible.Array;
        }

        #region Точное вычисление log-likelihood (для ма-а-ааленьких rbm)

        /// <summary>
        /// Точное вычисление log-likelihood (для ма-а-ааленьких rbm),
        /// но возвращается усредненный результат по переданному дата-сету
        /// </summary>
        /// <param name="u">Временный (только для расчета Z) провайдер OpenCL</param>
        /// <param name="data">Набор данных для расчета log-likelihood (по ним идет усреднение)</param>
        /// <param name="consoleLogEnabled">Разрешение логгинга в консоль</param>
        /// <returns>Значение log-likelihood</returns>
        public double CalculateExactLogLikelihood(
            CLProvider u,
            DataSet data,
            bool consoleLogEnabled = false)
        {
            #region validate

            if (u == null)
            {
                throw new  ArgumentNullException("u");
            }
            if (data == null || data.Count == 0)
            {
                throw new ArgumentNullException("data");
            }

            if (this.VisibleNeuronCount > 60) //чуть ниже лимита лонга
            {
                throw new InvalidOperationException("Слишком большая RBM #1");
            }

            if (this.HiddenNeuronCount > 26)
            {
                throw new InvalidOperationException("Слишком большая RBM #2");
            }

            #endregion

            var totalInputPixels = this.VisibleNeuronCount - 1; //visibleNeuronCountWithoutBias;
            long totalInputVariants = (long) Math.Pow(2, totalInputPixels);

            var totalHiddenPixels = this.HiddenNeuronCount - 1; //hiddenNeuronCountWithoutBias;
            int totalHiddenVariants = (int) Math.Pow(2, totalHiddenPixels);

            #region текст кернела, вычисляющего Z

            var zKernel = @"
typedef struct
{
    float VWeight[{0}];//with bias
} VisibleWeight;

typedef struct
{
    float HNeuron[{1}];//with bias
} HiddenNeuron;

__kernel void CalculateEnergy(
    __global HiddenNeuron * hidden,
    __global float * visible,
    __global VisibleWeight * weights,
    
    __global float * results,

    int hiddenNeuronCount,
    int visibleNeuronCount
    )
{
    int hz = get_global_id(0);

    float8 internalSummator8 = 0;

    for (int i8 = 0; i8 < visibleNeuronCount/8; i8++)
    {
        float8 vi8 = vload8(i8, visible);

        int hiddenMaxCount =
            (i8 * 8 + 7) == (visibleNeuronCount - 1) 
                ? (hiddenNeuronCount - 1) 
                : hiddenNeuronCount; //bias к bias отменяем

        for (int j = 0; j < hiddenMaxCount; j++)
        {
            float hj = hidden[hz].HNeuron[j];
            float8 wij8 = vload8(i8, weights[j].VWeight);
            float8 m8 = vi8 * hj * wij8;

            internalSummator8 -= m8;
        }
    }

    float internalSummator = 0;

    for (int i = visibleNeuronCount - visibleNeuronCount % 8; i < visibleNeuronCount; i++)
    //for (int i = 0; i < visibleNeuronCount; i++)
    {
        float vi = visible[i];

        int hiddenMaxCount =
            i == (visibleNeuronCount - 1) 
                ? (hiddenNeuronCount - 1) 
                : hiddenNeuronCount; //bias к bias отменяем

        for (int j = 0; j < hiddenMaxCount; j++)
        {
            float hj = hidden[hz].HNeuron[j];
            float wij = weights[j].VWeight[i];
            float m = vi * hj * wij;

            internalSummator -= m;
        }
    }

    results[hz] = 
        internalSummator8.s0 
        + internalSummator8.s1 
        + internalSummator8.s2 
        + internalSummator8.s3 
        + internalSummator8.s4
        + internalSummator8.s5 
        + internalSummator8.s6 
        + internalSummator8.s7 
        + internalSummator
        ;
}
";

            zKernel = zKernel.Replace("{0}", this.VisibleNeuronCount.ToString());
            zKernel = zKernel.Replace("{1}", this.HiddenNeuronCount.ToString());

            #endregion

            var calculateEnergy = u.CreateKernel(zKernel, "CalculateEnergy");

            //готовим массивы для Hidden (все возможные битовые варианты)
            var hMem = u.CreateFloatMem((totalHiddenPixels + 1) * totalHiddenVariants, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            for (long i = 0; i < totalHiddenVariants; i++)
            {
                //пробиваем биты текущего варианта
                for (var ii = 0; ii < totalHiddenPixels; ii++)
                {
                    long bit = (long) Math.Pow(2, ii);
                    long mask = i & bit;

                    hMem.Array[i*(totalHiddenPixels + 1) + ii] = mask > 0L ? 1f : 0f;
                }

                hMem.Array[i*(totalHiddenPixels + 1) + totalHiddenPixels] = 1f; //bias
            }
            hMem.Write(BlockModeEnum.Blocking);


            var iBatchSize = 10000; //сколько хидден вариантов просчитывать за один запуск кернела

            //заполняем видимое и результаты (размер батча)
            var vMemList = new List<OpenCL.Net.Wrapper.Mem.Mem<float>>();
            var hResultList = new List<OpenCL.Net.Wrapper.Mem.Mem<float>>();
            for (var cc = 0; cc < iBatchSize; cc++)
            {
                var vMem = u.CreateFloatMem(totalInputPixels + 1, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                vMem.Array[totalInputPixels] = 1f; //bias
                vMemList.Add(vMem);

                var hResult = u.CreateFloatMem(totalHiddenVariants, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                hResult.Write(BlockModeEnum.Blocking);
                hResultList.Add(hResult);
            }

            var beforeZ = DateTime.Now;

            //--------------------------------- вычисляем Z -------------------------------
            double z = 0f;

            for (long i = 0; i < totalInputVariants; i += iBatchSize)
            {
                var realBatchSize = Math.Min(iBatchSize, totalInputVariants - i);

                //заполняем визибл
                for (var cc = 0; cc < realBatchSize; cc++)
                {
                    var vMem = vMemList[cc];

                    for (var ii = 0; ii < totalInputPixels; ii++)
                    {
                        long bit = (long) Math.Pow(2, ii);
                        long mask = (i + cc) & bit;

                        vMem.Array[ii] = mask > 0L ? 1f : 0f;
                    }

                    vMem.Write(BlockModeEnum.NonBlocking);
                }

                u.QueueFinish();

                for (var cc = 0; cc < realBatchSize; cc++)
                {
                    var vMem = vMemList[cc];
                    var hResult = hResultList[cc];

                    //запускаем просчет Z для всех вариантов hidden
                    calculateEnergy
                        .SetKernelArgMem(0, hMem)
                        .SetKernelArgMem(1, vMem)
                        .SetKernelArgMem(2, this.Weights)
                        .SetKernelArgMem(3, hResult)
                        .SetKernelArg(4, 4, this.HiddenNeuronCount)
                        .SetKernelArg(5, 4, this.VisibleNeuronCount)
                        .EnqueueNDRangeKernel(totalHiddenVariants);
                }

                u.QueueFinish();

                //батч прочитали (размер батча * все возможные варианты скрытых нейронов)

                //читаем просчитанные варианты
                for (var cc = 0; cc < realBatchSize; cc++)
                {
                    hResultList[cc].Read(BlockModeEnum.NonBlocking);
                }

                u.QueueFinish();

                //"суммируем" прочитанные значения
                for (var cc = 0; cc < realBatchSize; cc++)
                {
                    foreach (var d in hResultList[cc].Array)
                    {
                        z += Math.Exp(-d);
                    }
                }

                if (consoleLogEnabled)
                {
                    Console.Write(
                        "{0}/{1} left {2}, {3}%         ",
                        i,
                        totalInputVariants,
                        (totalInputVariants - i),
                        (int) (i*100/totalInputVariants));
                    Console.SetCursorPosition(0, Console.CursorTop);
                }
            }

            var afterZ = DateTime.Now;
            var diffZ = afterZ - beforeZ;

            if (consoleLogEnabled)
            {
                Console.WriteLine(
                    "Z = {0}, ln(Z) = {1}, it takes {2}",
                    z,
                    Math.Log(z),
                    diffZ);
            }

            //------------ считаем положительную часть лог-лайклихуда (по переданным данным) -------
            var llList = new List<double>();

            var vMemPositive = vMemList[0]; //используем первый vMem от расчета Z (они больше не нужны)
            var hResultPositive = hResultList[0]; //используем первый hResult от расчета Z (они больше не нужны)

            for (var k = 1; k < data.Count; k++)
            {
                var vk = data[k].Input;

                vk.CopyTo(vMemPositive.Array, 0);
                vMemPositive.Write(BlockModeEnum.Blocking);

                //запускаем просчет Z для всех hidden
                calculateEnergy
                    .SetKernelArgMem(0, hMem) //используем от расчета Z, там он больше не нужен
                    .SetKernelArgMem(1, vMemPositive)
                    .SetKernelArgMem(2, this.Weights)
                    .SetKernelArgMem(3, hResultPositive)
                    .SetKernelArg(4, 4, this.HiddenNeuronCount)
                    .SetKernelArg(5, 4, this.VisibleNeuronCount)
                    .EnqueueNDRangeKernel(totalHiddenVariants);

                u.QueueFinish();

                hResultPositive.Read(BlockModeEnum.Blocking);

                var positive = 0.0;
                foreach (var d in hResultPositive.Array)
                {
                    positive += Math.Exp(-d);
                }

                llList.Add(Math.Log(positive) - Math.Log(z));
            }

            //усредняем LL по всем датапоинтам
            var avgLogLikelihood = llList.Average();

            if (consoleLogEnabled)
            {
                Console.WriteLine(
                    "Avg log likelihood = {0}",
                    avgLogLikelihood);
            }

            return
                avgLogLikelihood;
        }

        #endregion

        public Bitmap DrawGibbsChain(
            float[] start,
            int perImageGibbsSteps,
            int imageCount,
            int imageWidth,
            int imageHeight)
        {
            #region validate

            if (start == null || start.Length != (this.VisibleNeuronCount - 1))
            {
                throw new ArgumentException("start");
            }

            if (imageCount <= 0 || imageHeight <= 0 || imageWidth <= 0)
            {
                throw new ArgumentException("imageCount <= 0 || imageHeight <= 0 || imageWidth <= 0");
            }

            #endregion

            //картинка
            var q = (int)Math.Ceiling(Math.Sqrt(imageCount));
            var result = new Bitmap(
                q * imageWidth,
                q * imageHeight);

            //заполняем видимое
            Array.Copy(start, Visible.Array, VisibleNeuronCount - 1);
            Visible.Write(BlockModeEnum.Blocking);

            //рисуем начальную картинку
            CreateContrastEnhancedBitmapFromLayer(
                result,
                0,
                0,
                imageWidth,
                imageHeight,
                Visible.Array);

            for (var imageIndex = 1; imageIndex < imageCount; imageIndex++)
            {
                for (var gibbsIndex = 0; gibbsIndex < perImageGibbsSteps; gibbsIndex++)
                {
                    var randomIndex = Randomizer.Next(RandomCount);

                    SampleHidden
                        .SetKernelArgMem(0, Hidden0)
                        .SetKernelArgMem(1, Visible)

                        .SetKernelArgMem(2, Weights)
                        .SetKernelArgMem(3, Randoms)

                        .SetKernelArg(4, 4, HiddenNeuronCount)
                        .SetKernelArg(5, 4, VisibleNeuronCount)

                        .SetKernelArg(6, 4, randomIndex)
                        .SetKernelArg(7, 4, RandomCount)

                        .EnqueueNDRangeKernel(HiddenNeuronCount - 1); //without bias

                    //ComputeVisible
                    //    .SetKernelArgMem(0, Hidden0)
                    //    .SetKernelArgMem(1, Visible)

                    //    .SetKernelArgMem(2, Weights)

                    //    .SetKernelArg(3, 4, HiddenNeuronCount)
                    //    .SetKernelArg(4, 4, VisibleNeuronCount)

                    //    .EnqueueNDRangeKernel(VisibleNeuronCount - 1); //without bias

                    SampleVisible
                        .SetKernelArgMem(0, Hidden0)
                        .SetKernelArgMem(1, Visible)

                        .SetKernelArgMem(2, Weights)
                        .SetKernelArgMem(3, Randoms)

                        .SetKernelArg(4, 4, HiddenNeuronCount)
                        .SetKernelArg(5, 4, VisibleNeuronCount)

                        .SetKernelArg(6, 4, randomIndex)
                        .SetKernelArg(7, 4, RandomCount)

                        .EnqueueNDRangeKernel(VisibleNeuronCount - 1); //without bias
                }

                CLProvider.QueueFinish();
                
                Visible.Read(BlockModeEnum.Blocking);

                //рисуем картинку
                CreateContrastEnhancedBitmapFromLayer(
                    result,
                    (imageIndex % q) * imageWidth,
                    ((int)(imageIndex / q)) * imageHeight,
                    imageWidth,
                    imageHeight,
                    Visible.Array);
            }

            return result;
        }

        private void CreateContrastEnhancedBitmapFromLayer(
            Bitmap bitmap,
            int left,
            int top,
            int imageWidth,
            int imageHeight,
            float[] layer)
        {
            #region validate

            if (bitmap == null)
            {
                throw new ArgumentNullException("bitmap");
            }

            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            if (imageHeight <= 0 || imageWidth <= 0)
            {
                throw new ArgumentException("imageHeight <= 0 || imageWidth <= 0");
            }

            #endregion

            var max = layer.Take(imageWidth * imageHeight).Max(val => val);
            var min = layer.Take(imageWidth * imageHeight).Min(val => val);

            if (Math.Abs(min - max) <= float.Epsilon)
            {
                min = 0;
                max = 1;
            }

            for (int x = 0; x < imageWidth; x++)
            {
                for (int y = 0; y < imageHeight; y++)
                {
                    var value = layer[PointToIndex(x, y, imageWidth)];
                    value = (value - min) / (max - min);
                    var b = (byte)Math.Max(0, Math.Min(255, value * 255.0));

                    bitmap.SetPixel(left + x, top + y, Color.FromArgb(b, b, b));
                }
            }
        }

        private int PointToIndex(int x, int y, int width)
        {
            return y * width + x;
        }


        #region kernel source

        private string _kernelsSource = @"
typedef struct
{
    float VWeight[{0}];
} VisibleWeight;

__kernel void CalculateVisibleFreeEnergy(
    __global float * dataArray,
    __global VisibleWeight * weights,
    __global float * results,
    
    int hiddenCount, //with bias
    int visibleCount //with bias
    )
{
    int dataIndex = get_global_id(0);

    __global float* vd = dataArray + visibleCount * dataIndex;

    float4 vis4 = 0.0;
    for (int i4 = 0; i4 < (visibleCount - 1) / 4; i4++)
    {
        float4 vi4 = vload4(i4, vd);
        float4 ai4 = vload4(i4, weights[hiddenCount - 1].VWeight);

        vis4 += vi4 * ai4;
    }

    float vis = 0.0;
    for (int i = (visibleCount - 1) - ((visibleCount - 1) % 4); i < visibleCount - 1; i++)
    {
        float vi = vd[i];
        float ai = weights[hiddenCount - 1].VWeight[i];

        vis += vi * ai;
    }

    float hid = 0.0;
    for (int j = 0; j < hiddenCount - 1; j++)
    {
        float4 xj4 = 0.0;
        for (int i4 = 0; i4 < visibleCount / 4; i4++)
        {
            float4 vi4 = vload4(i4, vd);
            float4 wij4 = vload4(i4, weights[j].VWeight);

            xj4 += vi4 * wij4;
        }

        float xj = 0.0;
        for (int i = visibleCount - visibleCount % 4; i < visibleCount; i++)
        {
            float vi = vd[i];
            float wij = weights[j].VWeight[i];

            xj += vi * wij;
        }

        float expxj = exp(xj + xj4.s0 + xj4.s1 + xj4.s2 + xj4.s3);
        float hidPart = log(1 + expxj);

        hid += hidPart;
    }

    results[dataIndex] = -vis4.s0 - vis4.s1 - vis4.s2 - vis4.s3 -vis - hid; //freeEnergy of visible vector
}


__kernel void ErrorCompute(
    __global float * hidden0,
    __global float * input,

    __global float * hidden1,
    __global float * visible,

    __global VisibleWeight * nabla,

    int hiddenCount, //with bias
    int inputCount) //with bias
{
    int hiddenIndex = get_global_id(0);

    //задаем отрицательную часть изменения весов
    for (int inputIndex = 0; inputIndex < inputCount - 1; inputIndex++)
    {
        float error = 
            input[inputIndex] * hidden0[hiddenIndex]
            - visible[inputIndex] * hidden1[hiddenIndex];

        nabla[hiddenIndex].VWeight[inputIndex] += error;
    }
}

__kernel void SampleVisible(
    __global float * hidden,
    __global float * visible,

    __global VisibleWeight * weights,
    __global float * randoms,

    int hiddenCount, //with bias
    int visibleCount, //with bias

    int randomIndex,
    int randomCount)
{
    int visibleIndex = get_global_id(0);

    //высчитываем состояние скрытого нейрона
    float sum = 0;
    for (int hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex++)
    {
        sum += 
            weights[hiddenIndex].VWeight[visibleIndex]
            * hidden[hiddenIndex];
    }

    //вероятностное состояние нейрона
    float probability = 1.0 / (1.0 + exp(-sum));

    //уникальный рандом индекс для каждого work unit
    int wuri = randomIndex + visibleIndex;
    int correntRandomIndex = wuri % randomCount; //получение остатка (операция %) медленная на ГПУ (попробовать оптимизировать)

    //уникальный рандом для каждого work unit
    float random = randoms[correntRandomIndex];

    //вероятностное состояние нейрона
    visible[visibleIndex] = random <= probability ? 1 : 0;

}

__kernel void SampleHidden(
    __global float * hidden,
    __global float * visible,

    __global VisibleWeight * weights,
    __global float * randoms,

    int hiddenCount, //with bias
    int visibleCount, //with bias

    int randomIndex,
    int randomCount)
{
    int hiddenIndex = get_global_id(0);

    //высчитываем состояние скрытого нейрона
    int visibleIndex = 0;
    float4 sum4 = 0;
    for (visibleIndex = 0; visibleIndex < visibleCount / 4; visibleIndex++)
    {
        float4 vWeight4 = vload4(visibleIndex, weights[hiddenIndex].VWeight);
        float4 visible4 = vload4(visibleIndex, visible);

        sum4 += vWeight4 * visible4;
    }
    float sum = sum4.s0 + sum4.s1 + sum4.s2 + sum4.s3;
    for (visibleIndex = visibleIndex * 4; visibleIndex < visibleCount; visibleIndex++)
    {
        sum += weights[hiddenIndex].VWeight[visibleIndex] * visible[visibleIndex];
    }

    //уникальный рандом индекс для каждого work unit
    int wuri = randomIndex + hiddenIndex;
    int correntRandomIndex = wuri % randomCount; //получение остатка (операция %) медленная на ГПУ (попробовать оптимизировать)

    //уникальный рандом для каждого work unit
    float random = randoms[correntRandomIndex];

    //вероятностное состояние нейрона
    float probability = 1.0 / (1.0 + exp(-sum));
    hidden[hiddenIndex] = random <= probability ? 1 : 0;
}

__kernel void ComputeVisible(
    __global float * hidden,
    __global float * visible,

    __global VisibleWeight * weights,

    int hiddenCount, //with bias
    int visibleCount) //with bias
{
    int visibleIndex = get_global_id(0);

    //высчитываем состояние скрытого нейрона
    float sum = 0;
    for (int hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex++)
    {
        sum += weights[hiddenIndex].VWeight[visibleIndex] * hidden[hiddenIndex];
    }

    //вероятностное состояние нейрона
    float nstate = 1.0 / (1.0 + exp(-sum));
    visible[visibleIndex] = nstate;
}

__kernel void ComputeHidden(
    __global float * hidden,
    __global float * visible,

    __global VisibleWeight * weights,

    int hiddenCount, //with bias
    int visibleCount) //with bias
{
    int hiddenIndex = get_global_id(0);

    //высчитываем состояние скрытого нейрона
    int visibleIndex = 0;
    float4 sum4 = 0;
    for (visibleIndex = 0; visibleIndex < visibleCount / 4; visibleIndex++)
    {
        float4 vWeight4 = vload4(visibleIndex, weights[hiddenIndex].VWeight);
        float4 visible4 = vload4(visibleIndex, visible);

        sum4 += vWeight4 * visible4;
    }
    float sum = sum4.s0 + sum4.s1 + sum4.s2 + sum4.s3;
    for (visibleIndex = visibleIndex * 4; visibleIndex < visibleCount; visibleIndex++)
    {
        sum += weights[hiddenIndex].VWeight[visibleIndex] * visible[visibleIndex];
    }

    //состояние нейрона
    float nstate = 1.0 / (1.0 + exp(-sum));
    hidden[hiddenIndex] = nstate;

}

__kernel void ChangeWeight(
    __global float * weights,
    __global float * nabla,

    float learningRate)
{
    int weightIndex = get_global_id(0);

    weights[weightIndex] += learningRate * nabla[weightIndex];
}


__kernel void ChangeWeight4(
    __global float * currentLayerWeights,
    __global float * nabla,
    int count, //общее количество флоатов для обработки (для всех кернелов, длина currentLayerWeights, длина nabla)
    int kernelDataCount, //количество флоатов для обработки ОДНИМ кернелом (должно быть кратно 4м!!!)
    float learningRate)
{
    int kernelIndex = get_global_id(0);
    
    int d1StartIndex = kernelIndex * kernelDataCount;
    int d1Count = min(kernelDataCount, count - d1StartIndex);

    int d4StartIndex = d1StartIndex / 4;
    int d4Count = d1Count / 4;
    
    int d1StartRemainder = d1StartIndex + d4Count * 4;

    for(int cc = d4StartIndex; cc < d4StartIndex + d4Count; cc++)
    {
        float4 currentLayerWeights4 = vload4(cc, currentLayerWeights);
        float4 nabla4 = vload4(cc, nabla);

        float4 result = currentLayerWeights4 + (learningRate * nabla4);

        vstore4(
            result,
            cc,
            currentLayerWeights);
    }

    for(int cc = d1StartRemainder; cc < d1StartIndex + d1Count; cc++)
    {
        currentLayerWeights[cc] += learningRate * nabla[cc];
    }
}

__kernel void ClearKernel(
    __global float * nabla)
{
    int nablaIndex = get_global_id(0);

    nabla[nablaIndex] = 0;
}

__kernel void ClearKernel4(
    __global float * nabla,
    int count, //общее количество флоатов для обработки (для всех кернелов, длина currentLayerWeights, длина nabla)
    int kernelDataCount) //количество флоатов для обработки ОДНИМ кернелом (должно быть кратно 4м!!!)
{
    //__constant 
        float4 zero = 0;

    int kernelIndex = get_global_id(0);
    
    int d1StartIndex = kernelIndex * kernelDataCount;
    int d1Count = min(kernelDataCount, count - d1StartIndex);

    int d4StartIndex = d1StartIndex / 4;
    int d4Count = d1Count / 4;
    
    int d1StartRemainder = d1StartIndex + d4Count * 4;

    for(int cc = d4StartIndex; cc < d4StartIndex + d4Count; cc++)
    {
        vstore4(
            zero,
            cc,
            nabla);
    }

    for(int cc = d1StartRemainder; cc < d1StartIndex + d1Count; cc++)
    {
        nabla[cc] = 0;
    }
}

";

        #endregion

    }


}
