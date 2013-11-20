using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.Data.Visualizer
{
    public interface IVisualizer
    {
        /// <summary>
        /// Сохранить изображение квадратом
        /// </summary>
        /// <param name="filepath">Путь + имя файла</param>
        /// <param name="data">Данные</param>
        void SaveAsGrid(string filepath, List<float[]> data);

        /// <summary>
        /// Сохранить как список пар (например, оригинал и реконструкция)
        /// </summary>
        /// <param name="filepath">Путь + имя файла</param>
        /// <param name="data">Данные</param>
        void SaveAsPairList(string filepath, List<Pair<float[], float[]>> data);
    }
}
