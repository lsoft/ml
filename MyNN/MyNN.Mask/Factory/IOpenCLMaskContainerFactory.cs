namespace MyNN.Mask.Factory
{
    public interface IOpenCLMaskContainerFactory
    {
        IOpenCLMaskContainer CreateContainer(
            long arraySize,
            float p
            );
    }
}
