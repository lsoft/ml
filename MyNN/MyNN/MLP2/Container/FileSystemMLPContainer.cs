using System;
using System.IO;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Container
{
    public class FileSystemMLPContainer : IMLPContainer
    {
        private readonly string _rootFolder;
        private readonly ISerializationHelper _serializationHelper;

        public FileSystemMLPContainer(
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

        public void Save(
            IMLP mlp,
            IAccuracyRecord accuracyRecord)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (accuracyRecord == null)
            {
                throw new ArgumentNullException("accuracyRecord");
            }

            //var epocheRoot = Path.Combine(
            //    Name,
            //    accuracyRecord.EpocheNumber.ToString());
            //Directory.CreateDirectory(epocheRoot);

            //��������� ����
            var mlpFilePath = Path.Combine(_rootFolder, mlp.Name);
            _serializationHelper.SaveToFile(mlp, mlpFilePath);

            //��������� ���� � ������������
            var txtFilePath = Path.Combine(_rootFolder, "_result.txt");
            File.WriteAllText(txtFilePath, accuracyRecord.GetTextResults());
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

        public IMLPContainer GetChildContainer(string containerName)
        {
            if (containerName == null)
            {
                throw new ArgumentNullException("containerName");
            }

            var childRootFolder = Path.Combine(_rootFolder, containerName);

            return 
                new FileSystemMLPContainer(
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