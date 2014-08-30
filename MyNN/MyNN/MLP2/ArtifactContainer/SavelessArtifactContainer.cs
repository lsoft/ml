using System;
using System.IO;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.ArtifactContainer
{
    public class SavelessArtifactContainer : IArtifactContainer
    {
        private readonly string _rootFolder;
        private readonly ISerializationHelper _serializationHelper;

        public SavelessArtifactContainer(
            string rootFolder,
            ISerializationHelper serializationHelper
            )
        {
            if (rootFolder == null)
            {
                throw new ArgumentNullException("rootFolder");
            }
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }
            _rootFolder = rootFolder;
            _serializationHelper = serializationHelper;
        }

        public IMLP Load(string mlpName)
        {
            var path = Path.Combine(_rootFolder, mlpName);

            var result = _serializationHelper.LoadFromFile<MLP>(path);

            return result;
        }

        public IMLP DeepClone(IMLP mlp)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            return
                _serializationHelper.DeepClone(mlp);
        }

        public void SaveMLP(IMLP mlp, IAccuracyRecord accuracyRecord)
        {
            //nothing to do
        }

        public T LoadSerialized<T>(string resourceName)
        {
            if (resourceName == null)
            {
                throw new ArgumentNullException("resourceName");
            }

            var resourceFilePath = Path.Combine(_rootFolder, resourceName);

            return
                _serializationHelper.LoadFromFile<T>(resourceFilePath);
        }

        public void SaveSerialized<T>(T obj, string resourceName)
        {
            //nothing to do
        }

        public Stream GetWriteStreamForResource(string resourceName)
        {
            return
                Stream.Null;
        }

        public IArtifactContainer GetChildContainer(string containerName)
        {
            return 
                new SavelessArtifactContainer(
                    _rootFolder,
                    _serializationHelper
                    );
        }

        public void Clear()
        {
            //nothing to do
        }

        public void DeleteResource(string resourceName)
        {
            //nothing to do
        }
    }
}