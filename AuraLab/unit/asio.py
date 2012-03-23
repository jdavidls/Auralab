from unit._asio import *

# ASIO Channel Data Formats

UNKNOW_FORMAT = -1

INT16MSB = 0
INT24MSB  = 1		    # used for 20 bits as well
STINT32MSB = 2
STFLOAT32MSB = 3	    # IEEE 754 32 bit float
STFLOAT64MSB = 4	    # IEEE 754 64 bit double float

# these are used for 32 bit data buffer, with different alignment of the data inside
# 32 bit PCI bus systems can be more easily used with these
INT32MSB16 = 8		    # 32 bit data with 16 bit alignment
INT32MSB18 = 9		    # 32 bit data with 18 bit alignment
INT32MSB20 = 10 		# 32 bit data with 20 bit alignment
INT32MSB24 = 11	    	# 32 bit data with 24 bit alignment
	
INT16LSB = 16
INT24LSB = 17		    # used for 20 bits as well
INT32LSB = 18
FLOAT32LSB = 19		    # IEEE 754 32 bit float, as found on Intel x86 architecture
FLOAT64LSB = 20 		# IEEE 754 64 bit double float, as found on Intel x86 architecture

# these are used for 32 bit data buffer, with different alignment of the data inside
# 32 bit PCI bus systems can more easily used with these
INT32LSB16 = 24		    # 32 bit data with 18 bit alignment
INT32LSB18 = 25		    # 32 bit data with 18 bit alignment
INT32LSB20 = 26		    # 32 bit data with 20 bit alignment
INT32LSB24 = 27		    # 32 bit data with 24 bit alignment

#	ASIO DSD format.
INT8LSB1 = 32		    # DSD 1 bit data, 8 samples per byte. First sample in Least significant bit.
INT8MSB1 = 33		    # DSD 1 bit data, 8 samples per byte. First sample in Most significant bit.
INT8NER8 = 40           # DSD 8 bit data, 1 sample per byte. No Endianness required.




def enum_devices():
    import winreg
   
    asio_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "Software\\ASIO")

    i = -1
    while True:
        i += 1
        try:
            driver_name = winreg.EnumKey(asio_key, i)
        except WindowsError:
            winreg.CloseKey(asio_key)
            return

        try:
            driver_key = winreg.OpenKey(asio_key, driver_name)
        except WindowsError:
            continue

        ii = 0
        while True:
            try:
                name, value, type = winreg.EnumValue(driver_key, ii)
            except WindowsError:
                break

            if name == 'CLSID':
                yield (driver_name, value)
            ii += 1

        winreg.CloseKey(driver_key)
