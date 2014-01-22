using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.DeviceChooser
{
    public interface IDeviceChooser
    {
        void ChooseDevice(
            out Cl.DeviceType choosedDeviceType,
            out Cl.Device choosedDevice);
    }
}