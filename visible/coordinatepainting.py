import matplotlib.pyplot as plt
import random

color = ["or", "ob", "og", "ok"]
text_color = ["purple", "red", "blue", "green", "black", "yellow"]


def paintClusterResult(data, label, under_label):
    # 输出对应的纯度，RI，F1，熵的聚类准确度结果矢量图
    # 预计为条形图,data为[[]], 存放对应数据
    x = range(len(data[0]))
    x = [i * 2 for i in x]
    width = 0.3  # 方体宽度
    for i in range(len(data)):
        d = data[i]
        x_place = [p + width * i for p in x]  # 每个方体的x坐标
        rect = plt.bar(x=x_place, height=d, width=width, alpha=0.6, color=text_color[i % len(text_color)], label=label[i])
        for r in rect:
            height = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2, height+0.1, str(round(height, 3)), ha="center", va="bottom")
    plt.ylabel("Num")
    plt.ylim(0, 4)
    plt.xticks([i + width / 2 * (len(data) - 1) for i in x], under_label)
    plt.xlabel("Method")
    plt.title("Clustering Validation Method")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = [[0.4, 0.6, 0.3, 0.7, 0.1, 0.6], [0.8, 0.6, 0.6, 0.7, 0.5, 0.4],
            [0.2, 0.3, 0.2, 0.6, 0.6, 0.8], [0.8, 0.7, 0.6, 0.6, 0.3, 0.6]]
    label = ["method1", "method2", "method3", "method4"]
    paintClusterResult(data, label)