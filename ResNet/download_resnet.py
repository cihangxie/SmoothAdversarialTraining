import os, sys, getopt, gdown

usage_str = 'python download_resnet.py --model_name Res50-relu'
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
        elif arg == 'Res50-relu':
            ckpt_meta = 'https://drive.google.com/uc?id=132lC0k7EWjT371NZRY7OzULA-AshcDqc'
            ckpt_data = 'https://drive.google.com/uc?id=1f03RpPMF6_OyVnL7F8RnbkPT4uitnucX'
            ckpt_index = 'https://drive.google.com/uc?id=1cohW6FAE7dTaC7RARiqP3fVbiE5AVX4f'
        elif arg == 'Res50-silu':
            ckpt_meta = 'https://drive.google.com/uc?id=1NEDQyTYKTd4aw8lpWe4LsLC9EVf3Uat3'
            ckpt_data = 'https://drive.google.com/uc?id=1y3yAKrXgyGF7pnXoZg7R6Ge6N3LsXa_B'
            ckpt_index = 'https://drive.google.com/uc?id=1VvnMZyrntlLfWnQAdY9wfuCAGJ4fMU3x'
        elif arg == 'Res50-smoothrelu':
            ckpt_meta = 'https://drive.google.com/uc?id=1oibs4xXh6tZCezYaalJ4jyFFF55hzao7'
            ckpt_data = 'https://drive.google.com/uc?id=1mtm-QspciqS-_AXHNGgy-sqwtAAgtiw_'
            ckpt_index = 'https://drive.google.com/uc?id=18FizqadONHUAgysO8EoNyxFAzxpjJq_V'
        elif arg == 'Res50-gelu':
            ckpt_meta = 'https://drive.google.com/uc?id=1DzBg4N40f9Lm8hH9oLEIaocy-tOf5edW'
            ckpt_data = 'https://drive.google.com/uc?id=1bNkq7B_oq6AWoUQb74GlERRPTX8wQuFY'
            ckpt_index = 'https://drive.google.com/uc?id=1WXZNTAZwZ_JsgXN6mNwm3CMwyFTfknkb'
        elif arg == 'Res50-elu':
            ckpt_meta = 'https://drive.google.com/uc?id=12ebpFW5eJC0_V5GfwRf_7Xf4Epdqm4QZ'
            ckpt_data = 'https://drive.google.com/uc?id=1eymHm21mkxm5aL24Uca8aFqj6FPLqBcw'
            ckpt_index = 'https://drive.google.com/uc?id=11PCXlx7XNs7bkg2LH6RiCysJf5J2sgRq'
        elif arg == 'Res50-softplus':
            ckpt_meta = 'https://drive.google.com/uc?id=15kJvWfYNlgm3lIfKbQSfgV-bEyPoND8t'
            ckpt_data = 'https://drive.google.com/uc?id=16SLoDVJWK5w_69f8PoOHrxZZNakKaCXW'
            ckpt_index = 'https://drive.google.com/uc?id=1HnBSIWV4w8o2cD8fwb-WbHQZ5nxz7YSS'
        elif arg == 'Res50-mish':
            ckpt_meta = 'https://drive.google.com/uc?id=1I0ObFU__xnC2RLo99JVsHT29_JhwLIpA'
            ckpt_data = 'https://drive.google.com/uc?id=1yhHDRwbVA4okcGtJ4Y9pz0DbfvmH4zEg'
            ckpt_index = 'https://drive.google.com/uc?id=1W-4IR9IbJI-qBFlBfFlChp1OekjpQoGK'
        # differnet image resolution
        elif arg == 'Res50-silu-299':
            ckpt_meta = 'https://drive.google.com/uc?id=1PkArMefesGXqX7PGLI2ibKoMy31AndNK'
            ckpt_data = 'https://drive.google.com/uc?id=1Qlbet7k8_0RmHPzfaZvBpRv-c3R2cjCr'
            ckpt_index = 'https://drive.google.com/uc?id=12lpRIpz7hRN7bMHEkZAMqrZr2euirKOA'
        elif arg == 'Res50-silu-380':
            ckpt_meta = 'https://drive.google.com/uc?id=1AdjhY9FQa8dLA2Z5RT_3o9L570Nqh5Z_'
            ckpt_data = 'https://drive.google.com/uc?id=1tnHuq7t6PpyeSCeolEQHG7ZWKye4hAY6'
            ckpt_index = 'https://drive.google.com/uc?id=16bPLGP6Ltamtcjji2WV63RDxhRU2hoPU'
        # different width
        elif arg == 'ResX50-silu-32x4d':
            ckpt_meta = 'https://drive.google.com/uc?id=1oUh6B-0TynxRf7B-4INgDdKbvgzlihY7'
            ckpt_data = 'https://drive.google.com/uc?id=1jIMgBg5VjPp8mToRNvTpKs5jzUFbOceY'
            ckpt_index = 'https://drive.google.com/uc?id=1K4KHECxGM8H-QoY7N1XvS6_dxx6Ej87k'
        elif arg == 'ResX50-silu-32x8d':
            ckpt_meta = 'https://drive.google.com/uc?id=1c9vGl81tyaN6W-p3E_ogOw6ZJtorv4V4'
            ckpt_data = 'https://drive.google.com/uc?id=1_tbqY26jFp5TPIXDHQu1qyC2Rx_5ZrcW'
            ckpt_index = 'https://drive.google.com/uc?id=1-QZmB4eUDPQ2Pyx651gKI5CRamcOmBHk'
        # different depth
        elif arg == 'Res101-silu':
            ckpt_meta = 'https://drive.google.com/uc?id=1mSI36SYH2axrV_9vVeVdTFk0H9BWlLxn'
            ckpt_data = 'https://drive.google.com/uc?id=1r_IM4JXR1hmummpTbzHj1gb2BwrAZbi8'
            ckpt_index = 'https://drive.google.com/uc?id=1-95fYvBaCFoHWnM0USEOWeLORDdS48ek'
        elif arg == 'Res152-silu':
            ckpt_meta = 'https://drive.google.com/uc?id=1H1fMjSzgey41FbVmTbWXpbB0d8IVg8OD'
            ckpt_data = 'https://drive.google.com/uc?id=1D44_MDY0aylpHi7iJ3xqRHsQNCb-TFcX'
            ckpt_index = 'https://drive.google.com/uc?id=17V5SS8zeYEb-WEHk0_U_8T_20uujR3T9' 
        # compound scaling
        elif arg == 'ResX152-silu-380-32x8d':
            ckpt_meta = 'https://drive.google.com/uc?id=1hzFZ1BhBYdTlBuLK1tahww0hIKP-eZ4L'
            ckpt_data = 'https://drive.google.com/uc?id=1tpr3XLzM4s5HlQu7AloGnrMxDOFCDKAx'
            ckpt_index = 'https://drive.google.com/uc?id=1kk5QS8n8iVvisdtBu0-SyRR4WVRCYpE0'
        else:
            print('arch is not included in this repo. Must be one of {Res50-relu, Res50-silu, Res50-smoothrelu, Res50-gelu, Res50-elu, Res50-softplus, Res50-mish, Res50-silu-299, Res50-silu-380, ResX50-silu-32x4d, ResX50-silu-32x8d, Res101-silu, Res152-silu, ResX152-silu-380-32x8d}')

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
