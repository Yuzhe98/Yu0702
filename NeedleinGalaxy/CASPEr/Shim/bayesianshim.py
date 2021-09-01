import pyautogui
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
import serial


from bayes_opt import BayesianOptimization


ser = serial.Serial( timeout = 1, baudrate = 115200 )


# sio.flush() 
# hello = sio.readline()
s = ""

def Read():
    global s
    s = ""
    while True:
        c = ser.read(1)
        if c == b"\r":
            break
        elif  c == b"":
            # sys.stderr.write("not received data, probably lost data\n")
            return ""
        s = s + str(c)[2:-1]
    return s



def RampShims(channel, target):  

    global ser
    verbose = False
    ser.port = 'COM18'
    ser.open()

    if (verbose == True): print("ramping channel "+channel+":")

    ser.write(("GR"+str(channel)+"\r\n").encode("ascii")) #get rate
    Read()
    if (s != '1.000000'):
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

    if (abs(float(target)) > 3.0):
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

class ShimAppControl():
    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    F = 0
    G = 0
    H = 0
    def __init__(self, a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E = e
        self.F = f
        self.G = g
        self.H = h

    def applycurrent(self):
        for i in vars(ShimAppControl):
            if(len(i) == 1):
                RampShims(str(i), self.__dict__[i])

class NMRAppControl():
    def start1pulse(self):
        pyautogui.moveTo(423, 29)
        pyautogui.click(423, 29) # click KEA-NMR
        time.sleep(0.5)
        pyautogui.click(420, 100) # click "1pulse"
    def run(self):
        pyautogui.moveTo(554, 1189)
        # pyautogui.click(554, 1189) # click "KEA-NMR app"
        pyautogui.click(362, 947)  # click "run"
def Lorentzian(x, center, gamma, A, offset):
    return offset + 0.5*gamma*A / (np.pi * ((x-center)**2 + (0.5*gamma)**2))

# shim log file
sourceFile = open('shimming_record20210830.csv', 'a+')

def fit(expnumber):
    # make sure file is created and read
    waittime = 0
    while True:
        try:
            data = np.loadtxt('1pulse/'+str(expnumber)+'/data.csv', unpack=True, delimiter=',')      
        except:
            time.sleep(0.1)
            waittime = waittime + 1
            if waittime >= 10000:
                raise OSError("File " + '1pulse/'+str(expnumber)+"/data.csv not found!") 
        else:
            break
    FIDComplex = data[1] + 1j*data[2]

    spectrum = np.fft.fft(FIDComplex, norm='ortho')
    spectrum = np.abs(spectrum)**2

    frequencies = -np.fft.fftfreq(len(spectrum), d=0.1) # Set d to dwell time in ms

    popt, pcov = curve_fit(Lorentzian, frequencies, spectrum, p0=(frequencies[np.argmax(spectrum)], 0.1, 0.1, 0))
    plt.plot(frequencies, Lorentzian(frequencies, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
    plt.plot(frequencies, spectrum)
    # gamma equals FWHM for a Lorenztian
    plt.title('fit: center=%5.3f, linewidth=%5.3f, area=%5.3f, expnumber=%d' % ( popt[0], popt[1], popt[2], expnumber ) )
    plt.savefig('1pulse/'+str(expnumber)+'/fit.pdf')
    plt.close()
    plt.figure()
    # center, FWHM, area
    sourceFile.write('%5.3f, %5.3f, %5.3f\n' % ( popt[0], popt[1], popt[2] ))
    # area <= 0.1 means failed fit
    linewidth = popt[1] 
    if (popt[2] <= 0.05) or linewidth <= 0.005:
        linewidth = 1e100 
    return linewidth / popt[2]

initialexpnumber = 280 ##!! much check before use
print("Warning: starting with #"+str(initialexpnumber)+"!")

expnumber = initialexpnumber
# apply one pulse and return reciprocal of linewidth
def onepulse(a, b, c, d, e, f, g, h):
    global expnumber
   
    shimcurrent=ShimAppControl(a, b, c, d, e, f, g, h)
    shimcurrent.applycurrent()
    time.sleep(1)
    nmrobject = NMRAppControl()
    nmrobject.run()
    sourceFile.write('%d, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, ' % (expnumber, a, b, c, d, e, f, g, h))
    f = fit(expnumber)
    expnumber = expnumber + 1
    return 1/f



pbounds = {'a': (-2,3), 'b': (-2,3), 'c': (-2,3), 'd': (-2,3), 'e' : (-2,3), 'f' : (-2,3), 'g' : (-2,3), 'h' : (-2,3)}
# initialize optimizer
optimizer = BayesianOptimization(
    f = onepulse, # function that finds maximum
    pbounds = pbounds, # parameter boundaries
    verbose = 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1, 
)
# open port



# initial value
optimizer.probe( 							
    params={"a": -1.379, "b": 0.387, "c": -1.624, "d": 0.233, "e": 0.172, "f": -1.988, "g" : -1.539, "h" : -0.028},
    lazy=True,
)

optimizer.maximize(init_points=2, n_iter=200)



# plt.style.use('seaborn-colorblind')
# plt.rcParams['figure.figsize'] = [16, 9]
# plt.rcParams['font.size'] = 16
