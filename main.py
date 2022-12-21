from loraEnv import loraEnv
import numpy as np
def main():
    try:
        print('INICIO')
        myEnv = loraEnv(1)
        myEnv.step(5)
        print('FIN')

    except Exception as e:
        print(e)

# BER --> max: 0.00013895754823009532, min: 3.351781950877708e-07
# PDR --> max:1, min:1
# PRR --> max: 0.9999705047488826 , min:0.9878453578024127
# duration --> max: 15.635346635989748, min: 4.928511996677576
if __name__ == '__main__': main()