using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MyNN.Data.Visualizer
{
    public interface IVisualizer
    {
        /// <summary>
        /// Сохранить изображение квадратом
        /// </summary>
        /// <param name="writeStream">Поток для записи</param>
        /// <param name="data">Данные</param>
        void SaveAsGrid(Stream writeStream, List<float[]> data);

        /// <summary>
        /// Сохранить как список пар (например, оригинал и реконструкция)
        /// </summary>
        /// <param name="writeStream">Поток для записи</param>
        /// <param name="data">Данные</param>
        void SaveAsPairList(Stream writeStream, List<Pair<float[], float[]>> data);
    }
}
