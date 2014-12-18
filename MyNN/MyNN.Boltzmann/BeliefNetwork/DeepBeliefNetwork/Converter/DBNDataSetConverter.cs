using System;
using System.Collections.Generic;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;

namespace MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.Converter
{
    public class DBNDataSetConverter : IDataSetConverter
    {
        private readonly IContainer _container;
        private readonly IAlgorithm _algorithm;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IDataSetFactory _dataSetFactory;

        public DBNDataSetConverter(
            IContainer container,
            IAlgorithm algorithm,
            IDataItemFactory dataItemFactory,
            IDataSetFactory dataSetFactory
            )
        {
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }
            if (algorithm == null)
            {
                throw new ArgumentNullException("algorithm");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (dataSetFactory == null)
            {
                throw new ArgumentNullException("dataSetFactory");
            }

            _container = container;
            _algorithm = algorithm;
            _dataItemFactory = dataItemFactory;
            _dataSetFactory = dataSetFactory;
        }

        public IDataSet Convert(IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            var newdiList = new List<IDataItem>();
            foreach (var di in beforeTransformation)
            {
                _container.SetInput(di.Input);
                var nextLayer = _algorithm.CalculateHidden();

                var newdi = _dataItemFactory.CreateDataItem(
                    nextLayer,
                    di.Output);

                newdiList.Add(newdi);
            }

            var result = _dataSetFactory.CreateDataSet(
                new FromArrayDataItemLoader(
                    newdiList,
                    new DefaultNormalizer()),
                0
                );

            return result;
        }
    }
}