using System.Collections.Generic;
using System.Text;
using MyNN.Data;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.ForwardPropagation
{
    public interface IForwardPropagation
    {
        /// <summary>
        /// Сеть, на которую настроен просчетчик
        /// </summary>
        MLP MLP
        {
            get;
        }

        /// <summary>
        /// Получение значений на выходном слое сети
        /// </summary>
        /// <param name="dataSet">Данные для прохождения по сети</param>
        /// <returns>Значение выходного слоя</returns>
        List<ILayerState> ComputeOutput(DataSet dataSet);

        /// <summary>
        /// Вычисление состояние всей сети по одному примеру
        /// </summary>
        /// <param name="dataSet">Данные для прохождения по сети</param>
        /// <returns>Значение выходного слоя</returns>
        /// <returns>Состояние каждого нейрона сети для каждого примера</returns>
        List<IMLPState> ComputeState(DataSet dataSet);
    }
}
