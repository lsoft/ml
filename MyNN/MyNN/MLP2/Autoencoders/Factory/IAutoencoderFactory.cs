using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.BackpropagationFactory;
using MyNN.MLP2.ForwardPropagationFactory;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure.Layer;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.MLP2.Autoencoders.Factory
{
    //public interface IAutoencoderFactory
    //{
    //    IAutoencoder CreateAutoencoder(
    //        string root,
    //        string folderName,
    //        params LayerInfo[] layerInfos);

    //    IStackedAutoencoder CreateStackedAutoencoder(
    //        Func<DataSet, ITrainDataProvider> dataProviderFactory,
    //        Func<DataSet, IValidation> validationFactory,
    //        Func<int, ILearningAlgorithmConfig> configFactory,
    //        IBackpropagationAlgorithmFactory backpropagationAlgorithmFactory,
    //        IForwardPropagationFactory forwardPropagationFactory,
    //        params LayerInfo[] layerInfos);
    //}

    //class AutoencoderFactory : IAutoencoderFactory
    //{
    //    private readonly IDeviceChooser _deviceChooser;
    //    private readonly IRandomizer _randomizer;
    //    private readonly ISerializationHelper _serialization;

    //    public AutoencoderFactory(
    //        IDeviceChooser deviceChooser,
    //        IRandomizer randomizer,
    //        ISerializationHelper serialization)
    //    {
    //        if (deviceChooser == null)
    //        {
    //            throw new ArgumentNullException("deviceChooser");
    //        }
    //        if (randomizer == null)
    //        {
    //            throw new ArgumentNullException("randomizer");
    //        }
    //        if (serialization == null)
    //        {
    //            throw new ArgumentNullException("serialization");
    //        }

    //        _deviceChooser = deviceChooser;
    //        _randomizer = randomizer;
    //        _serialization = serialization;
    //    }

    //    public IAutoencoder CreateAutoencoder(
    //        string root,
    //        string folderName,
    //        params LayerInfo[] layerInfos)
    //    {
    //        //root, folderName  allowed to be null

    //        if (layerInfos == null)
    //        {
    //            throw new ArgumentNullException("layerInfos");
    //        }
    //        if (layerInfos.Length < 3)
    //        {
    //            throw new ArgumentException("layerInfos");
    //        }
    //        if (layerInfos.First().LayerSize != layerInfos.Last().LayerSize)
    //        {
    //            throw new ArgumentException("layerInfos sizes");
    //        }

    //        return 
    //            new Autoencoder(
    //                _randomizer,
    //                root,
    //                folderName,
    //                layerInfos);
    //    }

    //    public IStackedAutoencoder CreateStackedAutoencoder(
    //        Func<DataSet, ITrainDataProvider> dataProviderFactory,
    //        Func<DataSet, IValidation> validationFactory, 
    //        Func<int, ILearningAlgorithmConfig> configFactory, 
    //        IBackpropagationAlgorithmFactory backpropagationAlgorithmFactory,
    //        IForwardPropagationFactory forwardPropagationFactory,
    //        params LayerInfo[] layerInfos)
    //    {
    //        throw new NotImplementedException();
    //    }
    //}
}
