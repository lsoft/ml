using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.BoltzmannMachines.BinaryBinary.DBN;
using MyNN.MLP2.Structure.Layer.Factory.WeightLoader;
using MyNN.OutputConsole;

namespace MyNN.BoltzmannMachines.DBNInfo
{
    public class FileDBNInformation : IDBNInformation
    {
        private readonly string _dbnInfoRoot;
        private readonly ISerializationHelper _serializationHelper;

        /// <summary>
        /// Количество слоев в обученной DBN
        /// </summary>
        public int LayerCount
        {
            get
            {
                return
                    _layerSizes.Count;
            }
        }

        /// <summary>
        /// Размеры слоев обученной DBN
        /// </summary>
        private readonly List<int> _layerSizes;
        public int[] LayerSizes
        {
            get
            {
                return
                    _layerSizes.ToArray();
            }
        }

        public FileDBNInformation(
            string dbnInfoRoot,
            ISerializationHelper serializationHelper
            )
        {
            if (dbnInfoRoot == null)
            {
                throw new ArgumentNullException("dbnInfoRoot");
            }
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }

            _dbnInfoRoot = dbnInfoRoot;
            _serializationHelper = serializationHelper;

            ////количество слоев в обученной DBN
            //var dbnFolderCount =
            //    Directory.GetDirectories(dbnInfoRoot)
            //    .ToList()
            //    .FindAll(j => Path.GetFileName(j).StartsWith(DeepBeliefNetwork.RbmFolderName))
            //    .Count;

            #region проверяем наличие файла dbn.info и содержимое в нем

            var pathToDbnInfo = Path.Combine(dbnInfoRoot, "dbn.info");
            if (!File.Exists(pathToDbnInfo))
            {
                throw new InvalidOperationException("dbn.info does not found, information about MLP structure does not exits.");
            }

            ConsoleAmbientContext.Console.WriteLine("dbn.info found...");

            #endregion

            var layerSizes = ExtractLayersSizeFromDbnInfo(pathToDbnInfo);

            if (layerSizes == null || layerSizes.Count == 0)
            {
                throw new InvalidOperationException("layer sizes are empty, information about MLP structure does not exits.");
            }

            this._layerSizes = layerSizes;
        }

        /// <summary>
        /// Получить загрузчик весов для автоенкодера
        /// </summary>
        /// <param name="layerIndex">Индекс слоя в сети (начинается с 1)</param>
        /// <param name="mlpLayersCount">Общее количество слоев в сети</param>
        /// <param name="encoderWeightLoader">Загрузчик весов для слоя кодирования</param>
        /// <param name="decoderWeightLoader">Загрузчик весов для слоя декодирования</param>
        public void GetAutoencoderWeightLoaderForLayer(
            int layerIndex,
            int mlpLayersCount,
            out IWeightLoader encoderWeightLoader,
            out IWeightLoader decoderWeightLoader
            )
        {
            if (layerIndex <= 0)
            {
                throw new ArgumentException("Номер слоя должен начинаться с 1, так как для входного слоя веса не загружаются.");
            }

            var rbmFolderPath = Path.Combine(_dbnInfoRoot, DeepBeliefNetwork.RbmFolderName + (layerIndex - 1));

            var lastEpocheNumber =
                (from d in Directory.EnumerateDirectories(rbmFolderPath)
                 orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
                 select d).First();

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Processed layers {0}, {1} weights from: {2}",
                    layerIndex,
                    (mlpLayersCount - layerIndex),
                    lastEpocheNumber));

            var pathToWeightsFile = Path.Combine(lastEpocheNumber, "weights.bin"); //!!! заменить юзанием конст

            encoderWeightLoader = new RBMWeightLoader(pathToWeightsFile, _serializationHelper);
            decoderWeightLoader = new RBMAutoencoderWeightLoader(pathToWeightsFile, _serializationHelper);
        }

        /// <summary>
        /// Получить загрузчик весов для слоя сети
        /// </summary>
        /// <param name="layerIndex">Индекс слоя в сети (начинается с 1)</param>
        /// <param name="weightLoader">Загрузчик весов для слоя сети</param>
        public void GetWeightLoaderForLayer(
            int layerIndex,
            out IWeightLoader weightLoader)
        {
            if (layerIndex <= 0)
            {
                throw new ArgumentException("Номер слоя должен начинаться с 1, так как для входного слоя веса не загружаются.");
            }

            var rbmFolderPath = Path.Combine(_dbnInfoRoot, DeepBeliefNetwork.RbmFolderName + (layerIndex - 1));

            var lastEpocheNumber =
                (from d in Directory.EnumerateDirectories(rbmFolderPath)
                 orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
                 select d).First();

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Processed layers {0} weights from: {1}",
                    layerIndex,
                    lastEpocheNumber));

            var pathToWeightsFile = Path.Combine(lastEpocheNumber, "weights.bin"); //!!! заменить юзанием конст

            weightLoader = new RBMWeightLoader(pathToWeightsFile, _serializationHelper);
        }

        private int ExtractRBMEpocheNumberFromDirName(string dirname)
        {
            var lastSlashIndex = dirname.ToList().FindLastIndex(j => j == '\\');

            var stringNumber = dirname.Substring(lastSlashIndex + 8); //!!! константу 8 заменить на ссылку на имя.Length

            return int.Parse(stringNumber);
        }

        public static List<int> ExtractLayersSizeFromDbnInfo(string pathToDbnInfo)
        {
            if (!File.Exists(pathToDbnInfo))
            {
                return null;
            }

            var lines = File.ReadAllLines(pathToDbnInfo);
            if (lines.Length <= 1)
            {
                return null;
            }

            var layersLine = lines[1];
            if (string.IsNullOrEmpty(layersLine) || layersLine.IndexOf(':') < 0)
            {
                return null;
            }

            var layersString = layersLine.Substring(layersLine.IndexOf(':') + 1);
            if (string.IsNullOrEmpty(layersString) || !layersString.Contains('-'))
            {
                return null;
            }

            var result = new List<int>();

            var layersArray = layersString.Split(new[] { "-" }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var l in layersArray)
            {
                var size = 0;
                if (Int32.TryParse(l, out size))
                {
                    result.Add(size);
                }
                else
                {
                    result = null;
                    break;
                }
            }

            return result;
        }
    }
}
