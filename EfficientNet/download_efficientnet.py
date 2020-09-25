import os, sys, getopt, gdown

usage_str = 'python download_efficientnet.py --model_name efficientnet-b0'
def main(argv):
    model_name = None
    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'model_name='])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif arg == 'efficientnet-b0':
            ckpt_meta = 'https://drive.google.com/uc?id=1UgwnDU7kAlPd5l2Hp0AvKoYc-LD5jpAY'
            ckpt_data = 'https://drive.google.com/uc?id=1NVEtGO0l15r2Dc5FBzOKlWr2lqJ11WQD'
            ckpt_index = 'https://drive.google.com/uc?id=1ZTFo5RHQYy9zCKghFrQyACFQOsg4srr4'
        elif arg == 'efficientnet-b1':
            ckpt_meta = 'https://drive.google.com/uc?id=1izecRKPEWbZcaNnfAud7AS3REZldvHDQ'
            ckpt_data = 'https://drive.google.com/uc?id=12EWVkLi0_4Ep6Du933MV1vCwIKAActzT'
            ckpt_index = 'https://drive.google.com/uc?id=1Z5oKE9oYLMcl16uK_FOWjY_sbeFhljRp'
        elif arg == 'efficientnet-b2':
            ckpt_meta = 'https://drive.google.com/uc?id=1wgWGtZqOMq0LPR4FDelT1G64tbQu-pO0'
            ckpt_data = 'https://drive.google.com/uc?id=1QkDEm6OSgeeRwYd9suXtQbu-3yC9ebmu'
            ckpt_index = 'https://drive.google.com/uc?id=1-_GlLOlIhwnhu1lnT1rv9aleTR1VmXAs'
        elif arg == 'efficientnet-b3':
            ckpt_meta = 'https://drive.google.com/uc?id=1rq2Tx4qif5mQdY1e52usvv4LP4hb8gIb'
            ckpt_data = 'https://drive.google.com/uc?id=1zlCLPKaX4i9_ikuJzusY4aV9kTGzXLqF'
            ckpt_index = 'https://drive.google.com/uc?id=1BeyzyFYA43CwGvlXyNNG-2EswH4DLQNt'
        elif arg == 'efficientnet-b4':
            ckpt_meta = 'https://drive.google.com/uc?id=1XXrDvIILllqouDwygEI095arHhytvuY_'
            ckpt_data = 'https://drive.google.com/uc?id=17cMDXEqNjzXpPcvhSWAA7pDfTxGl3I9Q'
            ckpt_index = 'https://drive.google.com/uc?id=1O5N_WD-ofuSG5n5OOiNiZRpcBIhh0Ykg'
        elif arg == 'efficientnet-b5':
            ckpt_meta = 'https://drive.google.com/uc?id=1dOy6lnfrmPp861rLcl-pJ1s9wlgZi9rd'
            ckpt_data = 'https://drive.google.com/uc?id=1EWF0H_26zj6_kvdxQD1mYjjLzW9K7qHz'
            ckpt_index = 'https://drive.google.com/uc?id=1QG81MOXKF1MpUYV9nTECfOimmGCQuiH2'
        elif arg == 'efficientnet-b6':
            ckpt_meta = 'https://drive.google.com/uc?id=1x9wIeges7c9acxEV-ZrL1zIUkrqCQjmm'
            ckpt_data = 'https://drive.google.com/uc?id=1iA-XyCRmXzyvHr-IP21xvHxD2y4lesyK'
            ckpt_index = 'https://drive.google.com/uc?id=14y9VCC7mHSE6cX10hkof5-vFA2bkjd31'
        elif arg == 'efficientnet-b7':
            ckpt_meta = 'https://drive.google.com/uc?id=1l2iSHdISRu19WwBRMtv--wtEV_Ra6Zp8'
            ckpt_data = 'https://drive.google.com/uc?id=1koh9au89VJR6sB1b4ep2azbUmNiDmblj'
            ckpt_index = 'https://drive.google.com/uc?id=1SAZt2Ald7MVWsxz_Hpe7F1bTEZW7nQdF'
        elif arg == 'efficientnet-bL1':
            ckpt_meta = 'https://drive.google.com/uc?id=1xDI4Rg1r56gq9SKLg8DstezwLgKmMWfV'
            ckpt_data = 'https://drive.google.com/uc?id=1jGniWU77jQtjxjRqnd4GRoefCaJ9dnzr'
            ckpt_index = 'https://drive.google.com/uc?id=1MZr3Vx1E7qICamwpV_CFs1i2BZ40O6oq'
        elif arg == 'efficientnet-bL1-enhanced':
            ckpt_meta = 'https://drive.google.com/uc?id=1kepUwuVvENHT16ukAq6Wln_x2rVR2BEi'
            ckpt_data = 'https://drive.google.com/uc?id=1ceFsUZYWYh2usgVZV9uDnn9zc6NgWFty'
            ckpt_index = 'https://drive.google.com/uc?id=1CJvhOkKaEaPvNqe_ZZelsMzJpuitysAv'
        else:
            print('arch is not included in this repo. Must be one of {efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7, efficientnet-l1, efficientnet-l1-enhanced}')

    # download ckpt
    if os.path.isdir(arg):
        print('folder already exit')
        sys.exit()
    os.mkdir(arg)
    gdown.download(ckpt_meta, arg+'/'+arg+'.meta', quiet=False)
    gdown.download(ckpt_data, arg+'/'+arg+'.data-00000-of-00001', quiet=False)
    gdown.download(ckpt_index, arg+'/'+arg+'.index', quiet=False)
    print('files are saved at {}'.format(arg))

if __name__ == '__main__':
    main(sys.argv[1:])
