using System;
using System.Linq;


namespace OpenCL.Net.Wrapper.DeviceChooser
{
    public class IntelGPUDeviceChooser : IDeviceChooser
    {
        private readonly bool _showSelectedVendor;

        public IntelGPUDeviceChooser(bool showSelectedVendor = true)
        {
            _showSelectedVendor = showSelectedVendor;
        }

        public void ChooseDevice(
            out DeviceType choosedDeviceType,
            out Device choosedDevice)
        {
            ErrorCode error;

            var platforms = Cl.GetPlatformIDs(out error);
            if (error != ErrorCode.Success)
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Unable to retrieve an OpenCL Device, error was: {0}!",
                        error));
            }

            foreach (var platform in platforms)
            {
                var deviceIds = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error);
                if (deviceIds.Any())
                {
                    foreach (var device in deviceIds)
                    {
                        //var vendor = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out error).ToString();
                        //var uvendor = vendor.ToUpper();

                        //if (uvendor.Contains("INTEL"))
                        var vendorId = Cl.GetDeviceInfo(device, DeviceInfo.VendorId, out error).CastTo<int>();
                        if (vendorId == CLParameters.IntelVendorId)
                        {
                            if (_showSelectedVendor)
                            {
                                var deviceInfo = Cl.GetDeviceInfo(device, DeviceInfo.Name, out error).ToString();

                                Console.WriteLine(
                                    "Choosed device: {0}",
                                    deviceInfo);
                            }

                            choosedDevice = deviceIds.First();
                            choosedDeviceType = DeviceType.Gpu;
                            return;
                        }
                    }
                }
            }

            throw new InvalidOperationException("There is no NVIDIA/ATI/AMD GPU device");
        }
    }
}