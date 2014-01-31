using MyNN.MLP2.ForwardPropagation;

namespace MyNN.MLP2.Backpropagation.Validation
{
    public interface IValidation
    {
        /// <summary>
        /// Epoche validation
        /// </summary>
        /// <param name="forwardPropagation">Forward propagator</param>
        /// <param name="epocheRoot">Root epoche folder</param>
        /// <param name="allowToSave">Is it allowed to save MLP?</param>
        /// <returns>Current per-item error</returns>
        float Validate(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave);
    }
}
