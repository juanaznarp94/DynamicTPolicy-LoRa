from loraEnv import loraEnv
import numpy as np
def main():
    try:
        print('INICIO')
        myEnv = loraEnv(30, 4)
        myEnv.step(5)
        print('FIN')

    except Exception as e:
        print(e)


if __name__ == '__main__': main()