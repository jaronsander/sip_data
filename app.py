import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib import pyplot as plt, animation
import numpy as np

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

n_ch = 3

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

plt.axes(xlim=(0, 2), ylim=(-2, 2))
# fig = plt.Figure(dpi=100)
# ax = fig.add_subplot(xlim=(0, 2), ylim=(-1, 1))
# line, = ax.plot([], [], lw=2)
fig, ax = plt.subplots(nrows=n_ch, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button = tkinter.Button(master=root, text="Quit", command=root.quit)
button.pack(side=tkinter.BOTTOM)

sample_text = tkinter.Entry(root)
sample_text.pack()
def set_text_by_button():
    # Delete is going to erase anything
    # in the range of 0 and end of file,
    # The respective range given here
    sample_text.delete(0, "end")

    # Insert method inserts the text at
    # specified position, Here it is the
    # beginning
    sample_text.insert(0, "Text set by button")
ubutton = tkinter.Button(master=root, text="Update", command=set_text_by_button)
ubutton.pack(side=tkinter.RIGHT)

toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

# Do audio stuff




images = []
def init():
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x))
    for i in range(n_ch):
        images.append(ax[i].imshow)
    # line.set_data([], [])
    return ax,

def animate(i):
    print(i)
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    # if y[1] > 0.8:
    #     sample_text.insert(0, "True")
    # else:
    #     sample_text.insert(0, "True")
    #
    # line.set_data(x, y)
    # return line,
    for j in range(n_ch):
        ax[j].plot(x,y)
    return ax

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=200, interval=20, blit=False)

tkinter.mainloop()