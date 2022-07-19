import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main(module_name):

    if module_name == "synthesizer":
        # function to update the data
        loss_arr_193k = np.load(f"synthesizer_loss/synthesizer_loss_197k.npy")
        def my_function(i):
            # get data
            loss_arr = np.load("synthesizer_loss/synthesizer_loss.npy")
            # loss_arr = np.concatenate((loss_arr_193k, loss_arr), axis=0)
            # clear axis
            ax.cla()
            # plot cpu
            ax.plot(loss_arr)
            ax.scatter(len(loss_arr) - 1, loss_arr[-1])
            ax.text(len(loss_arr), loss_arr[-1], f"({len(loss_arr) - 1}, {loss_arr[-1]:.6})")
            plt.xlabel("Step-1")
            plt.ylabel("Loss")
            plt.title("Synthesizer Loss")
        # define and adjust figure
        fig, ax = plt.subplots()
        ax.set_facecolor('#DEDEDE')
        plt.xlabel("total steps")
        # animate
        ani = FuncAnimation(fig, my_function, interval=1000)
        plt.show()

    elif module_name == "vocoder":
        # function to update the data
        loss_arr_10k = np.load("vocoder_loss/vocoder_loss_10K.npy")
        loss_arr_30k = np.load("vocoder_loss/vocoder_loss_30K.npy")
        def my_function(i):
            # get data
            loss_arr = np.load("vocoder_loss/vocoder_loss.npy")
            # loss_arr = np.concatenate((loss_arr_10k, loss_arr_30k, loss_arr), axis=0)
            # clear axis
            ax.cla()
            # plot cpu
            ax.plot(loss_arr)
            ax.scatter(len(loss_arr) - 1, loss_arr[-1])
            ax.text(len(loss_arr), loss_arr[-1], f"({len(loss_arr) - 1}, {loss_arr[-1]:.6})")
            plt.xlabel("Step-1")
            plt.ylabel("Loss")
            plt.title("Vocoder Loss")
        # define and adjust figure
        fig, ax = plt.subplots()
        ax.set_facecolor('#DEDEDE')
        plt.xlabel("total steps")
        # animate
        ani = FuncAnimation(fig, my_function, interval=1000)
        plt.show()

main("vocoder")