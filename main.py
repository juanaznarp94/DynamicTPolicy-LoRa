from loraEnv import loraEnv
import numpy as np
def main():
    try:
        print('INICIO')
        myEnv = loraEnv(5)
        myEnv.step((1, 0))
        myEnv.reset()
        print('FIN')

    except Exception as e:
        print(e)

if __name__ == '__main__': main()