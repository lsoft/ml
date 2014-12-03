using System.Collections.Generic;
using System.IO;
using MyNN.Common.Other;

namespace MyNN.Common.Data.Visualizer
{
    public interface IVisualizer
    {
        ///// <summary>
        ///// Сохранить изображение квадратом
        ///// </summary>
        ///// <param name="writeStream">Поток для записи</param>
        ///// <param name="data">Данные</param>
        //void SaveAsGrid(Stream writeStream, List<float[]> data);

        ///// <summary>
        ///// Сохранить как список пар (например, оригинал и реконструкция)
        ///// </summary>
        ///// <param name="writeStream">Поток для записи</param>
        ///// <param name="data">Данные</param>
        //void SaveAsPairList(Stream writeStream, List<Pair<float[], float[]>> data);

        /// <summary>
        /// Визуализировать один итем квадратом
        /// </summary>
        /// <param name="data">Данные</param>
        void VisualizeGrid(float[] data);

        /// <summary>
        /// Визуализировать пару
        /// </summary>
        /// <param name="data">Данные</param>
        void VisualizePair(Pair<float[], float[]> data);

        /// <summary>
        /// Сохранить изображение квадратом
        /// </summary>
        /// <param name="writeStream">Поток для записи</param>
        void SaveGrid(Stream writeStream);

        /// <summary>
        /// Сохранить как список пар (например, оригинал и реконструкция)
        /// </summary>
        /// <param name="writeStream">Поток для записи</param>
        void SavePairs(Stream writeStream);

    }
}
