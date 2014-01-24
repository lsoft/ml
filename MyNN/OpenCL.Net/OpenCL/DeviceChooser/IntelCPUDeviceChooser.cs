using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL.DeviceChooser
{
    public class IntelCPUDeviceChooser : IDeviceChooser
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
                var deviceIds = Cl.GetDeviceIDs(platform, Cl.DeviceType.Cpu, out error);
                if (deviceIds.Any())
                {
                    foreach (var device in deviceIds)
                    {
                        var vendor = Cl.GetDeviceInfo(device, Cl.DeviceInfo.Vendor, out error).ToString();
                        if (vendor.ToUpper().Contains("INTEL"))
                        {
                            Console.WriteLine(
                                "Choosed vendor: {0}",
                                vendor.ToUpper());

                            choosedDevice = deviceIds.First();
                            choosedDeviceType = Cl.DeviceType.Cpu;
                            return;
                        }
                    }
                }
            }

            throw new InvalidOperationException("There is no Intel CPU device");
        }
    }
}
