import matplotlib.pyplot as plt
from utils import load_dict


filenames = ["BaseDeepFirstSearch",
             "DDQN_DeepFirstSearch",
             "pretrain_DDQN"]


def draw(x, y, linename=None, x_limit=int(1e6), y_limit=40):

    if not isinstance(x, list):
        x = list(x)
    if not isinstance(y, list):
        y = list(y)
    x.append(x_limit)
    y.append(y[-1])
    plt.plot(x, y, label=linename)


def draw_all():
    fig = plt.figure()
    for f in filenames:
        filename = "../results/" + f + ".json"
        dct = load_dict(filename)
        draw(dct["x"], dct["y"], f.replace("_", " "))
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Best Rewards")
    plt.show()


if __name__ == '__main__':
    draw_all()
