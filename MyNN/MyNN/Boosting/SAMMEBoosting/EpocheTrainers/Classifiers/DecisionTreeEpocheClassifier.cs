using System;
using Accord.MachineLearning.DecisionTrees;

namespace MyNN.Boosting.SAMMEBoosting.EpocheTrainers.Classifiers
{
    internal class DecisionTreeEpocheClassifier : IEpocheClassifier
    {
        private readonly DecisionTree _tree;

        public DecisionTreeEpocheClassifier(DecisionTree tree)
        {
            if (tree == null)
            {
                throw new ArgumentNullException("tree");
            }

            _tree = tree;
        }

        public int Compute(double[] input)
        {
            return
                _tree.Compute(input);
        }
    }
}