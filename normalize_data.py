import numpy as np                                                                                                             
import matplotlib.pyplot as plt
from scipy.stats import norm                                                  
import sys 

def func():               
    x = np.random.normal(4, 2, 10000)                                                            

    mean = np.mean(x)
    sigma = np.std(x)

    x -= mean # centering the data towards zero

    x_plot = np.linspace(min(x), max(x), 1000)                                                               

    fig = plt.figure()                                                               
    ax = fig.add_subplot(1,1,1)                                                      

    ax.hist(x, bins=50, normed=True, label="data")
    ax.plot(x_plot, norm.pdf(x_plot, mean, sigma), 'r-', label="pdf")                                                          

    ax.legend(loc='best')

    x_ticks = np.arange(-4*sigma, 4.1*sigma, sigma)                                  
    x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]                       

    ax.set_xticks(x_ticks)                                                           
    ax.set_xticklabels(x_labels)                                                     

    plt.show() 

def main(): 

    a = np.random.randint(1,10,(10,))

    with open('filename.txt', 'a') as f:
         print('This message will be written to a file.', file=f)
        #  print(f'{a}',file = f)

        #  write(f'{a}')
if __name__ == "__main__":

    main()
    