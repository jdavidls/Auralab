from time import sleep
from unit.asio import Driver, enum_devices
from unit.cuda import *

driver = Driver(clsid='{615C2C39-F8EB-11d3-B0B2-0000E8ED4AD9}')

print(driver.name)

#driver.controlPannel();

host = Host(device_id=0, synchronous=False)

driver.connect(host)

test = host.addUnit(Unit(host, "unit/waveform.ptx"));

driver.createBuffers((0,1), (0,1))

driver.play();

input();

driver.stop();

driver.destroyBuffers()

print(host.process_count)
input();