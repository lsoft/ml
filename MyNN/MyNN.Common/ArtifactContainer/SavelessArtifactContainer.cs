using System;
using System.IO;
using MyNN.Common.Other;

namespace MyNN.Common.ArtifactContainer
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

        public T DeepClone<T>(T obj)
        {
            return
                _serializationHelper.DeepClone(obj);
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