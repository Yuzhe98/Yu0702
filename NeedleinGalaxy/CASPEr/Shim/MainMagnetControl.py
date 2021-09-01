
from numpy import absolute, true_divide
import serial
import time
import math


ser = serial.Serial( timeout = 1, baudrate = 9600 )
s = ""


def read():
    global s
    s = ser.readline()
    s = str(s)
    s = s[2:-5]
    if (s == ''):
        time.sleep(0.05)
        read()
    else:
        print(s)
        return s


def StartMainCoil():

    global verbose
    verbose = True

    global ser
    ser.port = 'COM4'
    ser.open()

    ser.write("GS\r\n".encode("ascii")) #get current direction
    read()
    if (s != '........ CURRENT DIRECTION: POSITIVE'):
        ser.write("G!\r\n".encode("ascii")) #get maximum
        read()
        m = abs(float((s[28:-11])))
        ser.write(("S!"+str(m)+"\r\n").encode("ascii")) #set max to positive value
        ser.write("GS\r\n".encode("ascii"))

    ser.write("GT\r\n".encode("ascii")) #get tesla / amp
    read()
    if (s != '0.0172000'):
        ser.write("ST0.0172000\r\n".encode("ascii")) #set tesla / amp
        read()

    ser.write("GL\r\n".encode("ascii")) #get limit 
    read()
    if (s != '5.0'):
        ser.write("SL5.0\r\n".encode("ascii")) # set limit
        read() 

    ser.write("H\r\n".encode("ascii")) #get heater output
    read()
    if (s != '2.5'):
        ser.write("SH2.5\r\n".encode("ascii")) # set heater output
        read()

    ser.write("T1\r\n".encode("ascii")) #use tesla as unit
        # command has no output

    ser.write("S!0.1001\r\n".encode("ascii")) #set max to 0.1001T
    ser.write("G!\r\n".encode("ascii")) #check max
    read()
    if (s != '0.1001'):
        ser.write("S!0.1001\r\n".encode("ascii"))
        ser.write("G!\r\n".encode("ascii"))
        read()

    ser.write("XTRIP0\r\n".encode("ascii")) #turn external trip off
    #no output?

    ser.write("GH\r\n".encode("ascii")) #get heater status
    read()

    ser.write("SR0.1\r\n".encode("ascii")) #set ramp rate to 0.1 A/s
    ser.write("RS\r\n".encode("ascii")) #check ramp status
    read()
    if (s != '0.100 A/SEC'):
        ser.write("SR0.10000\r\n".encode("ascii"))
        ser.write("RS\r\n".encode("ascii"))
        read()

    ser.write("GO\r\n".encode("ascii")) #get output
    read() #returns "8.63784E-T at 0.0V"

    ser.close()



def RampMainCoil(target):

    global verbose
    verbose = True

    global ser
    ser.port = 'COM4'
    ser.open()

    ser.write("GO\r\n".encode("ascii")) #get output
    read()

    ser.write("RS\r\n".encode("ascii")) #ramp status
    read()

    ser.write("H\r\n".encode("ascii")) #heater status
    read()

    ser.write("GS\r\n".encode("ascii")) #get current sign, should always be +
    read()

    ser.write("H1\r\n".encode("ascii")) #turn heater on
    time.sleep(10) #wait 10 seconds
    ser.write("GH\r\n".encode("ascii")) #heater status
    read()
    if (s != '........ HEATER STATUS: ON'):
        print("heater wasn't on, turning on now")
        ser.write("H1\r\n".encode("ascii"))
        time.sleep(10)
        ser.write("H\r\n".encode("ascii"))  

    if (float(target) > 0.1001):
        target = 0.1001
        print("Max Current is 0.1001")
    ser.write(("S%"+str(target)+"\r\n").encode("ascii")) #set mid  (target)
    ser.write("G%\r\n".encode("ascii")) #get mid
    read()
    if (s != str(target)):
        ser.write(("S%"+str(target)+"\r\n").encode("ascii"))
        ser.write("G%\r\n".encode("ascii"))
        read()

    ser.write("R%\r\n".encode("ascii")) #ramp to mid
        #no output

    ser.write("P0\r\n".encode("ascii")) #make sure pause is off

    ser.write("GO\r\n".encode("ascii"))

    ser.write("RS\r\n".encode("ascii"))

    ser.write("H0\r\n".encode("ascii")) #turn heater off
    time.sleep(10)
    ser.write("GH\r\n".encode("ascii")) #heater status
    read()
    if (s != '........ HEATER STATUS: OFF'):
        print("heater wasn't off, turning off now")
        ser.write("H0\r\n".encode("ascii"))
        time.sleep(10)
        ser.write("H\r\n".encode("ascii"))  

    ser.write("SR0.10000\r\n".encode("ascii")) #set ramp rate to 0.1 A/s
    ser.write("GR\r\n".encode("ascii")) 
    read()
    if (s != '0.10000'):
        ser.write("SR0.10000\r\n".encode("ascii"))
        ser.write("GR\r\n".encode("ascii"))
        read()
    ser.write("R0\r\n".encode("ascii")) #ramp to zero
        #no output

    ser.write("P0\r\n".encode("ascii"))

    ser.write("GO\r\n".encode("ascii"))

    ser.write("RS\r\n".encode("ascii"))

    ser.close()



StartMainCoil()
RampMainCoil(0.1)