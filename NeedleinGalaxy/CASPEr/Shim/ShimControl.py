
from numpy import absolute, true_divide
import serial
import time
import math
import sys


ser = serial.Serial( timeout = 1, baudrate = 115200 )
s = ""

def Read():
    global s
    s = ""
    while True:
        c = ser.read(1)
        if c == b"\r":
            break
        elif  c == b"":
            print("not received data, probably lost data")
            print("current data is "+ repr(s))
            break
        s = s + str(c)[2:-1]
    return s
# def Read():
#     global s
#     s = ser.readline()
#     s = str(s)
#     s = s[2:-3]
#     if (s == ''):
#         time.sleep(0.05)
#         Read()
#     else:
#         return s


def RampShims(channel, target):  

    global verbose
    verbose = True

    global ser 
    ser.port = 'COM18'
    ser.open()

    if (verbose == True): print("ramping channel "+channel+":")

    ser.write(("GR"+str(channel)+"\r\n").encode("ascii")) #get rate
    Read()
    if (s != '1.0'):
        ser.write(("SR"+str(channel)+"1.0\r\n").encode("ascii")) #set rate
        ser.write(("GR"+str(channel)+"\r\n").encode("ascii"))
        Read()
        if (verbose == True): print("channel "+channel+" ramp rate: "+s+"A/s")

    ser.write("Q0\r\n".encode("ascii")) #reset quench

    ser.write(("H"+str(channel)+"1\r\n").encode("ascii")) #turn heater on

    ser.write(("CH"+str(channel)+"\r\n").encode("ascii")) #check heater
    Read()
    if (s != 'Y'):
        ser.write(("H"+str(channel)+"1\r\n").encode("ascii"))
    if (verbose == True): print("channel "+channel+" heater status: "+s)   

    if (float(target) > 3.0):
        target = 3.0
        print("invalid target current. max current is 3A")
    ser.write(("S%"+str(channel)+str(target)+"\r\n").encode("ascii")) #set target
    ser.write(("G%"+str(channel)+"\r\n").encode("ascii")) #get target
    Read()
    if (s != str(target)):
        ser.write(("S%"+str(channel)+str(target)+"\r\n").encode("ascii"))
        ser.write(("G%"+str(channel)+"\r\n").encode("ascii"))
        Read()
    if (verbose == True): print("channel "+channel+" target current: "+s+"A")

    ser.write(("R%"+str(channel)+str(target)+"\r\n").encode("ascii")) #ramp to target
        #no output
    if (verbose == True): print("now ramping...")

    ser.write(("CT"+str(channel)+"1\r\n").encode("ascii")) #check if channel is at target
    Read() #returns Y/N
    if (s != 'Y'):
        ser.write(("R%"+str(channel)+str(target)+"\r\n").encode("ascii"))
        ser.write(("CT"+str(channel)+"1\r\n").encode("ascii"))
        Read()
    if (verbose == True): print("channel "+channel+" reached target? "+s)
    
    ser.write(("GQ"+str(channel)+"1\r\n").encode("ascii")) #get quench
    Read()
    if (verbose == True): print("channel "+channel+" quench status: "+s)

    ser.close()



RampShims("A", 0)
RampShims("B", 0)
RampShims("C", 0)
RampShims("D", 0)
RampShims("E", 0)
RampShims("F", 0) 
RampShims("G", 0)
RampShims("H", 0)


# ser.port = 'COM18'
# ser.open()
# ser.write(("HA0\r\n").encode("ascii"))
# ser.write(("HB0\r\n").encode("ascii"))
# ser.write(("HC0\r\n").encode("ascii"))
# ser.write(("HD0\r\n").encode("ascii"))
# ser.write(("HE0\r\n").encode("ascii"))
# ser.write(("HF0\r\n").encode("ascii"))
# ser.write(("HG0\r\n").encode("ascii"))
# ser.write(("HH0\r\n").encode("ascii"))