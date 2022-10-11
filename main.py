from loraEnv import loraEnv
import sac
def main():
    try:
        print('INICIO')
        myEnv = loraEnv(15)
        myEnv.step(4)
        print('FIN')

    except Exception as e:
        print(e)


if __name__ == '__main__': main()

