using System;
using MyNN.Common.OutputConsole;
using MyNN.MLP.DBNInfo;
using MyNN.MLP.DBNInfo.WeightLoader;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Factory
{
    public class MLPFactory : IMLPFactory
    {
        private readonly ILayerFactory _layerFactory;

        public MLPFactory(
            ILayerFactory layerFactory
            )
        {
            if (layerFactory == null)
            {
                throw new ArgumentNullException("layerFactory");
            }

            _layerFactory = layerFactory;
        }

        public IMLP CreateMLP(
            string name,
            IFunction[] activationFunction, 
            params int[] neuronCountList)
        {
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }


            if (neuronCountList == null)
            {
                throw new ArgumentNullException("neuronCountList");
            }
            if (activationFunction == null || activationFunction.Length != neuronCountList.Length)
            {
                throw new InvalidOperationException("activationFunction == null || activationFunction.Length != neuronCountList.Length");
            }

            //формируем слои
            var layerList = new ILayer[neuronCountList.Length];

            //создаем входной слой
            layerList[0] = _layerFactory.CreateInputLayer(neuronCountList[0]);

            //создаем скрытые слои и выходной слой
            var isPreviousLayerHadBiasNeuron = true;
            for (var cc = 1; cc < neuronCountList.Length; cc++)
            {
                var isLayerHasBiasNeuron = cc != (neuronCountList.Length - 1);

                layerList[cc] = _layerFactory.CreateLayer(
                    activationFunction[cc],
                    neuronCountList[cc],
                    neuronCountList[cc - 1],
                    isLayerHasBiasNeuron,
                    isPreviousLayerHadBiasNeuron
                    );

                isPreviousLayerHadBiasNeuron = isLayerHasBiasNeuron;
            }

            return
                new MLP(
                    name,
                    _layerFactory,
                    layerList);
        }

        public IMLP CreateMLP(
            string name,
            ILayer[] layerList
            )
        {
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }
            if (layerList == null)
            {
                throw new ArgumentNullException("layerList");
            }

            return
                new MLP(
                    name,
                    _layerFactory,
                    layerList);
        }

        public IMLP CreateMLP(
            IDBNInformation dbnInformation,
            string name,
            IFunction[] activationFunction,
            params int[] neuronCountList)
        {
            if (dbnInformation == null)
            {
                throw new ArgumentNullException("dbnInformation");
            }
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }


            if (neuronCountList == null)
            {
                throw new ArgumentNullException("neuronCountList");
            }
            if (activationFunction == null || activationFunction.Length != neuronCountList.Length)
            {
                throw new InvalidOperationException("activationFunction == null || activationFunction.Length != neuronCountList.Length");
            }


            #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

            if (activationFunction.Length < dbnInformation.LayerCount)
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Layer count from dbn.info {0} != layer count from parameters {1}",
                        dbnInformation.LayerCount,
                        activationFunction.Length));
            }

            #endregion

            //формируем слои
            var layerList = new ILayer[neuronCountList.Length];

            //создаем входной слой
            layerList[0] = _layerFactory.CreateInputLayer(neuronCountList[0]);

            //создаем скрытые слои и выходной слой
            var isPreviousLayerHadBiasNeuron = true;
            for (var layerIndex = 1; layerIndex < neuronCountList.Length; layerIndex++)
            {
                var isLayerHasBiasNeuron = layerIndex != (neuronCountList.Length - 1);

                var layer = _layerFactory.CreateLayer(
                    activationFunction[layerIndex],
                    neuronCountList[layerIndex],
                    neuronCountList[layerIndex - 1],
                    isLayerHasBiasNeuron,
                    isPreviousLayerHadBiasNeuron
                    );

                if (layerIndex < dbnInformation.LayerCount)
                {
                    //загружаем веса
                    IWeightLoader weightLoader;
                    dbnInformation.GetWeightLoaderForLayer(layerIndex, out weightLoader);
                    weightLoader.LoadWeights(layer);
                }

                layerList[layerIndex] = layer;
                isPreviousLayerHadBiasNeuron = isLayerHasBiasNeuron;
            }

            return
                new MLP(
                    name,
                    _layerFactory,
                    layerList);
        }

        public IMLP CreateMLP(
            IDBNInformation dbnInformation,
            string name,
            IFunction[] activationFunction
            )
        {
            if (dbnInformation == null)
            {
                throw new ArgumentNullException("dbnInformation");
            }
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

            if (activationFunction.Length < dbnInformation.LayerCount)
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Layer count from dbn.info {0} != layer count from parameters {1}",
                        dbnInformation.LayerCount,
                        activationFunction.Length));
            }

            #endregion

            #region создаем слои

            ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

            var layerList = new ILayer[activationFunction.Length];

            layerList[0] = _layerFactory.CreateInputLayer(dbnInformation.LayerSizes[0]);

            for (var layerIndex = 1; layerIndex <= Math.Min(layerList.Length, dbnInformation.LayerCount); layerIndex++)
            {
                var isLayerHasBiasNeuron = layerIndex != (dbnInformation.LayerCount - 1);

                //создаем слой
                var layer = _layerFactory.CreateLayer(
                    activationFunction[layerIndex],
                    dbnInformation.LayerSizes[layerIndex],
                    layerList[layerIndex - 1].NonBiasNeuronCount,
                    isLayerHasBiasNeuron,
                    layerList[layerIndex - 1].IsBiasNeuronExists
                    );

                if (dbnInformation.LayerCount < layerIndex)
                {
                    //загружаем веса
                    IWeightLoader weightLoader;
                    dbnInformation.GetWeightLoaderForLayer(layerIndex, out weightLoader);
                    weightLoader.LoadWeights(layer);
                }

                layerList[layerIndex] = layer;
            }

            ConsoleAmbientContext.Console.WriteLine("Load weights done");

            #endregion

            //создаем MLP
            var mlp =
                new MLP(
                    name,
                    _layerFactory,
                    layerList);

            return
                mlp;
        }

        public IMLP CreateAutoencoderMLP(
            IDBNInformation dbnInformation,
            string name,
            IFunction[] activationFunction
            )
        {
            if (dbnInformation == null)
            {
                throw new ArgumentNullException("dbnInformation");
            }
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

            //проверяем что количество слоев в DBN и количество слоев в MLP сочетаемо
            var mlpLayersCount = activationFunction.Length;
            var autoencoderLayers = (mlpLayersCount - 1) / 2;
            if (dbnInformation.LayerCount - 1 != autoencoderLayers)
            {
                throw new InvalidOperationException("layer count from dbn.info != count of activation functions");
            }

            #endregion

            #region создаем слои

            ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

            var layerList = new ILayer[activationFunction.Length];

            layerList[0] = _layerFactory.CreateInputLayer(dbnInformation.LayerSizes[0]);

            for (var layerIndex = 1; layerIndex <= Math.Min(autoencoderLayers, dbnInformation.LayerCount); layerIndex++)
            {
                IWeightLoader encoderWeightLoader, decoderWeightLoader;
                dbnInformation.GetAutoencoderWeightLoaderForLayer(
                    layerIndex,
                    mlpLayersCount,
                    out encoderWeightLoader,
                    out decoderWeightLoader);

                //создаем слой кодирования
                {
                    var encoderLayer = _layerFactory.CreateLayer(
                        activationFunction[layerIndex],
                        dbnInformation.LayerSizes[layerIndex],
                        layerList[layerIndex - 1].NonBiasNeuronCount,
                        true,
                        layerList[layerIndex - 1].IsBiasNeuronExists
                        );

                    //загружаем веса
                    encoderWeightLoader.LoadWeights(encoderLayer);

                    layerList[layerIndex] = encoderLayer;
                }

                //создаем слой декодирования
                {
                    var decoderLayerIndex = mlpLayersCount - layerIndex;
                    var isDecoderLayerHasBiasNeuron = decoderLayerIndex != (activationFunction.Length - 1);

                    var decoderLayer = _layerFactory.CreateLayer(
                        activationFunction[decoderLayerIndex],
                        layerList[layerIndex - 1].NonBiasNeuronCount,
                        layerList[layerIndex].NonBiasNeuronCount,
                        isDecoderLayerHasBiasNeuron,
                        true
                        );

                    //загружаем веса
                    decoderWeightLoader.LoadWeights(decoderLayer);

                    layerList[decoderLayerIndex] = decoderLayer;
                }
            }

            ConsoleAmbientContext.Console.WriteLine("Load weights done");

            #endregion

            //создаем MLP
            var mlp =
                new MLP(
                    name,
                    _layerFactory,
                    layerList);

            return
                mlp;
        }

    }
}