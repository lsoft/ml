using System;
using System.Collections.Generic;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using MyNN.Boosting.SAMMEBoosting.EpocheTrainers.Classifiers;

namespace MyNN.Boosting.SAMMEBoosting.EpocheTrainers
{
    public class C45DecisionTreeTrainer : IEpocheTrainer
    {
        private readonly int _maxTreeHeight;
        private Random _rnd;

        public C45DecisionTreeTrainer(int maxTreeHeight)
        {
            _maxTreeHeight = maxTreeHeight;
            _rnd = new Random(DateTime.Now.Millisecond);
        }

        public IEpocheClassifier TrainEpocheClassifier(
            List<double[]> inputEpocheInputs,
            List<int> epocheLabels,
            int outputLength,
            int inputLength)
        {
            #region случайно сгенерируем номера столбцов, которые затрем нулями

            var zeroColumnIndexes = new List<int>();
            for (var cc = 0; cc < inputLength; cc++)
            {
                if (_rnd.NextDouble() < 0.5)
                {
                    zeroColumnIndexes.Add(cc);
                }
            }

            //не больше половины столбцов
            if (zeroColumnIndexes.Count > inputLength / 2)
            {
                zeroColumnIndexes = zeroColumnIndexes.Take(inputLength / 2).ToList();
            }

            #endregion

            #region клонируем данный нам свыше датасет и обнуляем столбцы

            var epocheInputs = new List<double[]>();
            foreach (var i in inputEpocheInputs)
            {
                var cloned = new double[i.Length];
                i.CopyTo(cloned, 0);

                foreach (var z in zeroColumnIndexes)
                {
                    cloned[z] = 0.0;
                }

                epocheInputs.Add(cloned);
            }

            #endregion

            //строим классификатор
            var attributes = new DecisionVariable[inputLength];
            for (var cc = 0; cc < attributes.Length; cc++)
            {
                attributes[cc] = new DecisionVariable("cc" + cc, DecisionVariableKind.Continuous);
            }

            // Create the Decision tree
            var tree = new DecisionTree(attributes, outputLength);

            // Creates a new instance of the C4.5 learning algorithm
            var c45 = new C45Learning(tree);

            if (this._maxTreeHeight > 0)
            {
                c45.MaxHeight = this._maxTreeHeight;
            }

            // Learn the decision tree
            var treeError = c45.Run(
                epocheInputs.ToArray(),
                epocheLabels.ToArray());

            //var acode = tree.ToCode("A");

            Console.WriteLine("Tree error: " + treeError);

            return
                new DecisionTreeEpocheClassifier(tree);
        }
    }
}