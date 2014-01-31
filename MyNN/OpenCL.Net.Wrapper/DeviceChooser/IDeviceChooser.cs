

namespace OpenCL.Net.Wrapper.DeviceChooser
{
    public interface IDeviceChooser
    {
        void ChooseDevice(
            out DeviceType choosedDeviceType,
            out Device choosedDevice);
    }
}