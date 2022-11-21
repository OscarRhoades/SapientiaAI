import numpy as np
import matplotlib.pyplot as plt


def show_array(H,range):
    K = H.cpu().detach().numpy()
    
    plt.imshow(K[0])

    plt.colorbar(orientation='vertical')
    plt.clim(0, range) 
    plt.show()
    exit()
