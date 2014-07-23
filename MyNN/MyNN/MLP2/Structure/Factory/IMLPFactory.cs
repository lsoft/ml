using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.BoltzmannMachines.DBNInfo;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Factory
{
    public interface IMLPFactory
    {
        IMLP CreateMLP(
            string name,
            IFunction[] activationFunction,
            params int[] neuronCountList
            );

        IMLP CreateMLP(
            string name,
            ILayer[] layerList
            );

        /// <summary>
        /// Создаем MLP по структуре предобученной DBN
        /// </summary>
        IMLP CreateMLP(
            IDBNInformation dbnInformation,
            string name,
            IFunction[] activationFunction
            );

        ///// <summary>
        ///// Создаем MLP по структуре предобученной DBN
        ///// </summary>
        ///// <param name="dbnInfoRoot">Путь к папке, где обучена DBN</param>
        //IMLP CreateMLP(
        //    string dbnInfoRoot,
        //    string root,
        //    string folderName,
        //    IFunction[] activationFunction
        //    );

        /// <summary>
        /// Создаем автоенкодер-MLP по структуре предобученной DBN
        /// </summary>
        IMLP CreateAutoencoderMLP(
            IDBNInformation dbnInformation,
            string name,
            IFunction[] activationFunction
            );

        ///// <summary>
        ///// Создаем автоенкодер-MLP по структуре предобученной DBN
        ///// </summary>
        ///// <param name="dbnInfoRoot">Путь к папке, где обучена DBN</param>
        //IMLP CreateAutoencoderMLP(
        //    string dbnInfoRoot,
        //    string root,
        //    string folderName,
        //    IFunction[] activationFunction
        //    );
    }
}
