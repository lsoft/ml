using System;
using System.Linq;


namespace OpenCL.Net.Wrapper.DeviceChooser
{
    public class IntelCPUDeviceChooser : IDeviceChooser
    {
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
                var deviceIds = Cl.GetDeviceIDs(platform, DeviceType.Cpu, out error);
                if (deviceIds.Any())
                {
                    foreach (var device in deviceIds)
                    {
                        var vendor = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out error).ToString();
                        if (vendor.ToUpper().Contains("INTEL"))
                        {
                            Console.WriteLine(
                                "Choosed vendor: {0}",
                                vendor.ToUpper());

                            choosedDevice = deviceIds.First();
                            choosedDeviceType = DeviceType.Cpu;
                            return;
                        }
                    }
                }
            }

            throw new InvalidOperationException("There is no Intel CPU device");
        }
    }
}
