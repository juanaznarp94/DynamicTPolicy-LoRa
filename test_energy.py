import numpy as np
import matplotlib.pyplot as plt

T = 600  # s
BW = 125  # KHz
PACKET_SIZE_BITS = 26 * 8  # Bits
PACKET_SIZE_BYTES = PACKET_SIZE_BITS / 8

CAPACITY = 13  # Ah
VOLTAGE = 3.6  # V
CUT_OFF_VOLTAGE = 2.2  # V


def battery_life():
    """
    standard battery solution(s) for your project
    Project parameters
    Environment: Smart City
    Application: Air quality monitoring
    Location: Outdoor
    Geographic zone: Europe
    Connectivity solution: LoRa
    Data transmission: Every 10'
    Life duration: 5 years
    Cut-off Voltage: 2,2 V
    Consumption profile
    Maximal peak current: 50 mA
    Yearly consumption: 1,29 Ah
    Total consumption: 6.45 Ah
    Recommended batteries:
        -LM 26500
        -LSH 20
        -LSP 26500-20F
    :return:
    """

    t_tx = 1.5  # seconds
    a_tx = 45 * 1e-3  # A

    t_rx = 0.54  # seconds
    a_rx = 15.2 * 1e-3  # A

    t_idle = 1.27  # seconds
    a_idle = 3 * 1e-3  # A

    t_sleep = T - (t_idle + t_rx + t_tx)  # seconds, calculate sleep time by substracting busy values to T
    a_sleep = 14 * 1e-6  # A

    c_tx = a_tx * t_tx / 3600  # Ah
    c_rx = a_rx * t_rx / 3600  # Ah
    c_idle = a_idle * t_idle / 3600  # Ah
    c_sleep = a_sleep * t_sleep / 3600  # Ah

    c_total = c_rx + c_tx + c_idle + c_sleep

    yearly = c_total * 6 * 24 * 365  # Ah yearly consumption
    # We send a packet 6 times per hour, and a year have 24*365 hours

    battery = 13 * (
                2.2 / 3.6)  # calculate the fraction of 13.000 mAh used having into account cutoff and voltage values
    duration = battery / yearly  # divide the max battery available by the yearly amount, so we will obtain years
    print(duration, "years")


years = battery_life()
