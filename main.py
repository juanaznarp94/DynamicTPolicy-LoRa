from loraEnv import loraEnv
import numpy as np
def main():
    try:
        print('INICIO')
        myEnv = loraEnv(10)
        myEnv.step(1)
        print('FIN')

    except Exception as e:
        print(e)

if __name__ == '__main__': main()