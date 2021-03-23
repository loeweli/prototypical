import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'dataaug':
        ds = protonets.data.omniglot.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
