namespace MyNN.KNN
{
    public interface IKNearest
    {
        int Classify(
            float[] itemToClassify,
            int knn);
    }
}