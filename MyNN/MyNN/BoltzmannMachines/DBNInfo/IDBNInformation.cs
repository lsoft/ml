using System.Collections.Generic;
using MyNN.MLP2.Structure.Layer.Factory.WeightLoader;

namespace MyNN.BoltzmannMachines.DBNInfo
{
    public interface IDBNInformation
    {
        /// <summary>
        /// ���������� ����� � ��������� DBN
        /// </summary>
        int LayerCount
        {
            get;
        }

        /// <summary>
        /// ������� ����� ��������� DBN
        /// </summary>
        int[] LayerSizes
        {
            get;
        }

        /// <summary>
        /// �������� ��������� ����� ��� ������������
        /// </summary>
        /// <param name="layerIndex">������ ���� � ���� (���������� � 1)</param>
        /// <param name="mlpLayersCount">����� ���������� ����� � ����</param>
        /// <param name="encoderWeightLoader">��������� ����� ��� ���� �����������</param>
        /// <param name="decoderWeightLoader">��������� ����� ��� ���� �������������</param>
        void GetAutoencoderWeightLoaderForLayer(
            int layerIndex,
            int mlpLayersCount,
            out IWeightLoader encoderWeightLoader,
            out IWeightLoader decoderWeightLoader
            );

        /// <summary>
        /// �������� ��������� ����� ��� ���� ����
        /// </summary>
        /// <param name="layerIndex">������ ���� � ���� (���������� � 1)</param>
        /// <param name="weightLoader">��������� ����� ��� ���� ����</param>
        void GetWeightLoaderForLayer(
            int layerIndex,
            out IWeightLoader weightLoader);
    }
}