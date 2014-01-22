using System;
using System.Linq;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.DeviceChooser
{
    public class NvidiaGPUDeviceChooser : IDeviceChooser
    {
        public void ChooseDevice(
            out Cl.DeviceType choosedDeviceType,
            out Cl.Device choosedDevice)
        {
            Cl.ErrorCode error;

            var platforms = Cl.GetPlatformIDs(out error);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Unable to retrieve an OpenCL Device, error was: {0}!",
                        error));
            }

            foreach (var platform in platforms)
            {
                var deviceIds = Cl.GetDeviceIDs(platform, Cl.DeviceType.Gpu, out error);
                if (deviceIds.Any())
                {
                    foreach (var device in deviceIds)
                    {
                        var vendor = Cl.GetDeviceInfo(device, Cl.DeviceInfo.Vendor, out error).ToString();
                        if (vendor.ToUpper().Contains("NVIDIA"))
                        {
                            choosedDevice = deviceIds.First();
                            choosedDeviceType = Cl.DeviceType.Gpu;
                            return;
                        }
                    }
                }
            }

            throw new InvalidOperationException("There is no NVIDIA GPU device");
        }
    }
}