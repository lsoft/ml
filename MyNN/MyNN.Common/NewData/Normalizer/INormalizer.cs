namespace MyNN.Common.NewData.Normalizer
{
    public interface INormalizer
    {
        /// <summary>
        /// Линейная нормализация [0;1]
        /// </summary>
        void Normalize(
            float[] dataToNormalize,
            float bias = 0f
            );

        /// <summary>
        /// Гауссова нормализация
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        void GNormalize(
            float[] dataToNormalize
            );
    }
}