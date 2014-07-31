using System;
using System.IO;
using System.Reflection;

namespace OpenCL.Net.Wrapper.Resource
{
    public class EmbeddedResourceReader
    {
        public EmbeddedResourceReader()
        {
        }

        public string GetTextResourceFile(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();

            return
                GetTextResourceFile(assembly, resourceName);
        }

        public string GetTextResourceFile(
            Assembly assembly,
            string resourceName)
        {
            if (assembly == null)
            {
                throw new ArgumentNullException("assembly");
            }
            if (resourceName == null)
            {
                throw new ArgumentNullException("resourceName");
            }

            var stream = assembly.GetManifestResourceStream(resourceName);
            if (stream != null)
            {
                try
                {
                    using (var reader = new StreamReader(stream))
                    {
                        var result = reader.ReadToEnd();
                        return result;
                    }
                }
                finally
                {
                    stream.Dispose();
                }
            }

            return null;

        }
    }
}