import numpy as np
import matplotlib.pyplot as plt
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)
if __name__ == "__main__":
    train_path = r"test_preacc.txt"  # 存储文件路径

    y_train = data_read(train_path)  # 训练准确率值，即y轴
    x_train = range(len(y_train))  # 训练阶段准确率的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epochs')  # x轴标签
plt.ylabel('Precision')  # y轴标签

# 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 增加参数color='red',这是红色。
plt.plot(x_train, y_train, color='red', linewidth=1, linestyle="solid", label="test")
plt.legend()
plt.title('Precision curve')
plt.show()
