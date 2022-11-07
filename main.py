from loraEnv import loraEnv
def main():
    try:
        print('INICIO')
        myEnv = loraEnv(20)
        myEnv.step(5)
        print('FIN')

    except Exception as e:
        print(e)


if __name__ == '__main__': main()

