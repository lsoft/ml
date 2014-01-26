using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Saver
{
    public class FileSystemMLPSaver : IMLPSaver
    {
        private readonly ISerializationHelper _serializationHelper;

        public FileSystemMLPSaver(
            ISerializationHelper serializationHelper)
        {
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }
            _serializationHelper = serializationHelper;
        }

        public void Save(
            string epocheRoot,
            IAccuracyRecord accuracyRecord,
            MLP mlp)
        {
            if (epocheRoot == null)
            {
                throw new ArgumentNullException("epocheRoot");
            }
            if (accuracyRecord == null)
            {
                throw new ArgumentNullException("accuracyRecord");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            string networkFilename;
            if (
                accuracyRecord.CorrectCount != int.MinValue
                && accuracyRecord.TotalCount != int.MinValue
                && accuracyRecord.ValidationPerItemError <= float.MinValue)
            {
                networkFilename = string.Format(
                        "{0}-{1} correct out of {2} - {3}%.mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        accuracyRecord.CorrectCount,
                        accuracyRecord.TotalCount,
                        accuracyRecord.CorrectPercent);
            }
            else if (
                accuracyRecord.CorrectCount != int.MinValue
                && accuracyRecord.TotalCount != int.MinValue
                && accuracyRecord.ValidationPerItemError > float.MinValue)
            {
                networkFilename = string.Format(
                        "{0}-({1}) {2} correct out of {3} - {4}%.mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        accuracyRecord.ValidationPerItemError,
                        accuracyRecord.CorrectCount,
                        accuracyRecord.TotalCount,
                        accuracyRecord.CorrectPercent);
            }
            else if (
                accuracyRecord.CorrectCount == int.MinValue
                && accuracyRecord.TotalCount == int.MinValue
                && accuracyRecord.ValidationPerItemError > float.MinValue)
            {
                networkFilename = string.Format(
                        "{0}-({1}).mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        accuracyRecord.ValidationPerItemError);
            }
            else
            {
                networkFilename = string.Format(
                        "{0}.mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"));
            }

            _serializationHelper.SaveToFile(
                mlp,
                Path.Combine(epocheRoot, networkFilename));
        }
    }
}
