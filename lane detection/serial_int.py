#!/usr/bin/python

#from __future__ import print_function
import serial 
import sys
import time
import threading


class GetHWPoller(threading.Thread):
  
  def __init__(self,sleeptime,pollfunc):

    self.sleeptime = sleeptime
    self.pollfunc = pollfunc  
    threading.Thread.__init__(self)
    self.runflag = threading.Event()  # clear this to pause thread
    self.runflag.clear()
    # response is a byte string, not a string
    self.response = b''
    
  def run(self):
    self.runflag.set()
    self.worker()

  def worker(self):
    while(1):
      if self.runflag.is_set():
        self.pollfunc()
   

  def pause(self):
    self.runflag.clear()

  def resume(self):
    self.runflag.set()

  def running(self):
    return(self.runflag.is_set())


class HW_Interface(object):
  """Class to interface with asynchrounous serial hardware.
  Repeatedly polls hardware, unless we are sending a command
  "ser" is a serial port class from the serial module """

  def __init__(self,ser,sleeptime):
    self.ser = ser
    self.sleeptime = float(sleeptime)
    self.worker = GetHWPoller(self.sleeptime,self.poll_HW)
    self.worker.setDaemon(True)
    self.response = None # last response retrieved by polling
    self.worker.start()
    self.callback = None
    self.verbose = True # for debugging

  def register_callback(self,proc):
    """Call this function when the hardware sends us serial data"""
    self.callback = proc
    #self.callback("test!")
    
  def kill(self):
    self.worker.kill()


  def write_HW(self,command):
    """ Send a command to the hardware"""
    self.ser.write(command)
    self.ser.flush()


    
  def poll_HW(self):
    """Called repeatedly by thread. Check for interlock, if OK read HW
    Stores response in self.response, returns a status code, "OK" if so"""

    response = self.ser.readline()
    if response is not None:
      if len(response) > 0: # did something write to us?
        response = response.strip() #get rid of newline, whitespace
        if len(response) > 0: # if an actual character
          if self.verbose:
            self.response = response
            data = self.response.decode("utf-8")
            #print("poll response: " + data)
            data = data.split(':')
            if data[0]=='data':
              x = float(data[1])
              y = float(data[2])
              v = float(data[3])
              print(x)
              print(y)
              print(v)
            sys.stdout.flush()
          #if self.callback:
          #a valid response so convert to string and call back
          #self.callback(self.response.decode("utf-8"))
        return "OK"
    return "None" # got no response


def my_callback(response):
  """example callback function to use with HW_interface class.
     Called when the target sends a byte, just print it out"""
  #print('got HW response "%s"' % response)



if __name__ == '__main__':
  """ You can run this from the command line to test"""

  # Need to set the portname for your Arduino serial port:
  # see "Serial Port" entry in the Arduino "Tools" menu
  ser = serial.Serial('/dev/ttyUSB0', 115200)

  sys.stdout.flush()
  hw = HW_Interface(ser,0.1)
  
  #when class gets data from the Arduino, it will call the my_callback function
  hw.register_callback(my_callback)

  while(1):
    ser.reset_input_buffer()
    sys.stdout.flush()
    cmd = input('--> ')
    #cmd = cmd.split()
    sys.stdout.flush()
    #if cmd[0] == 'r':
    #  print('Last response: "%s"' % hw.response)

    #if cmd[0] == 'M':
      #M = "M" +str(cmd)
    ser.write(cmd.encode('utf_8'))
    
    sys.stdout.flush()

    if cmd[0] == 'x':
      print("exiting...")
      exit()
