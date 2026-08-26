import matplotlib.pyplot as plt

#matplotlib.rcParams.update({'font.size': 22})


def set_up_graph(MEDIUM_SIZE=70, SMALLER_SIZE=30):
    BIGGER_SIZE = 35
    plt.figure(figsize=(40, 20))

    #plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('axes', titlesize=MEDIUM_SIZE)

