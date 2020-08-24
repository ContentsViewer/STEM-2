import serial
import time

def clear_sensor_input(ser: serial):
    ser.reset_input_buffer()
    ser.readline()

def read_latest_line(ser: serial):
    clear_sensor_input(ser)
    return ser.readline()

def begin_serial(port, baudrate) -> serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.parity = serial.PARITY_ODD
    ser.open()
    ser.close()
    ser.parity = serial.PARITY_NONE
    ser.open()
    time.sleep(2)
    return ser

