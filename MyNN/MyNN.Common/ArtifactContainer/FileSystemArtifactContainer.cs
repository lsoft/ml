using System;
using System.IO;
using MyNN.Common.Other;

namespace MyNN.Common.ArtifactContainer
{
    public class FileSystemArtifactContainer : IArtifactContainer
    {
        private readonly string _rootFolder;
        private readonly ISerializationHelper _serializationHelper;

        public FileSystemArtifactContainer(
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
            

            if (!Directory.Exists(_rootFolder))
            {
                Directory.CreateDirectory(_rootFolder);
            }
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

        public void SaveString(string message, string resourceName)
        {
            if (resourceName == null)
            {
                throw new ArgumentNullException("resourceName");
            }

            var resourceFilePath = Path.Combine(_rootFolder, resourceName);

            File.WriteAllText(
                resourceFilePath,
                message
                );
        }

        public void SaveSerialized<T>(T obj, string resourceName)
        {
            if (resourceName == null)
            {
                throw new ArgumentNullException("resourceName");
            }

            var resourceFilePath = Path.Combine(_rootFolder, resourceName);

            _serializationHelper.SaveToFile(obj, resourceFilePath);
        }

        public Stream GetWriteStreamForResource(string resourceName)
        {
            if (resourceName == null)
            {
                throw new ArgumentNullException("resourceName");
            }

            var resourceFilePath = Path.Combine(_rootFolder, resourceName);


            FileStream result = null;

            if (!File.Exists(resourceFilePath))
            {
                result = File.Create(
                    resourceFilePath);
            }
            else
            {
                result=  new FileStream(
                    resourceFilePath,
                    FileMode.Append,
                    FileAccess.Write);
            }

            return result;

        }

        public IArtifactContainer GetChildContainer(string containerName)
        {
            if (containerName == null)
            {
                throw new ArgumentNullException("containerName");
            }

            var childRootFolder = Path.Combine(_rootFolder, containerName);

            return 
                new FileSystemArtifactContainer(
                    childRootFolder,
                    _serializationHelper);
        }

        public void Clear()
        {
            Directory.Delete(_rootFolder, true);
            Directory.CreateDirectory(_rootFolder);
        }

        public void DeleteResource(string resourceName)
        {
            if (resourceName == null)
            {
                throw new ArgumentNullException("resourceName");
            }

            var resourceFilePath = Path.Combine(_rootFolder, resourceName);

            if (File.Exists(resourceFilePath))
            {
                File.Delete(resourceFilePath);
            }
        }
    }
}