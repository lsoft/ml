using System;
using System.Collections.Generic;
using MyNN.Common.Data;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation
{
    /// <summary>
    /// MLP forward propagation interface
    /// </summary>
    public interface IForwardPropagation
    {
        /// <summary>
        /// Сеть, на которую настроен просчетчик
        /// </summary>
        IMLP MLP
        {
            get;
        }

        /// <summary>
        /// Получение значений на выходном слое сети
        /// </summary>
        /// <param name="dataItemList">Данные для прохождения по сети</param>
        /// <returns>Значение выходного слоя</returns>
        List<ILayerState> ComputeOutput(IEnumerable<IDataItem> dataItemList);

        /// <summary>
        /// Получение значений на выходном слое сети
        /// </summary>
        /// <param name="dataItemList">Данные для прохождения по сети</param>
        /// <param name="propagationTime">Время просчета (без учета времени подготовки)</param>
        /// <returns>Значение выходного слоя</returns>
        List<ILayerState> ComputeOutput(IEnumerable<IDataItem> dataItemList, out TimeSpan propagationTime);

        /// <summary>
        /// Вычисление состояние всей сети по одному примеру
        /// </summary>
        /// <param name="dataItemList">Данные для прохождения по сети</param>
        /// <returns>Значение выходного слоя</returns>
        /// <returns>Состояние каждого нейрона сети для каждого примера</returns>
        List<IMLPState> ComputeState(IEnumerable<IDataItem> dataItemList);
    }
}
