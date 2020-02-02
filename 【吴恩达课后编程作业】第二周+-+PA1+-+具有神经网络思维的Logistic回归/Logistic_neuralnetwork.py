import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


m_train = train_set_y.shape[1]  # 训练集里图片的数量。
m_test = test_set_y.shape[1]  # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

# 现在看一看我们加载的东西的具体情况
print("训练集的数量: m_train = " + str(m_train))
print("测试集的数量 : m_test = " + str(m_test))
print("每张图片的宽/高 : num_px = " + str(num_px))
print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集_图片的维数: " + str(test_set_x_orig.shape))
print("测试集_标签的维数: " + str(test_set_y.shape))

# 将训练集的维度降低并转置。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    return 1/(1+np.exp(-z))


def predict(w , b , X ):
    """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据

    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

    """

    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0

    # 使用断言
    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def main():

    alpha = 0.005  # alpha is the learning rate
    X = train_set_x  # input
    Y = train_set_y  # training set's ground_truth
    X_test = test_set_x
    Y_test = test_set_y

    m = X.shape[1]

    w = np.zeros(shape = (X.shape[0], 1))
    b = 0
    num_iterations = 1500

    costs = []

    # starting train for num_iterations:
    for i in range(num_iterations):

        # propagate
        A = sigmoid(np.dot(w.T, X)+b)
        cost = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

        # back propagate
        dw = (1 / m) * np.dot(X, (A-Y).T)
        db = (1 / m) *np.sum(A-Y)

        cost = np.squeeze(cost)

        '''
        # create a dictionary to save the "dw,db"
        grads = {
                "dw": dw,
                "db": db
                }
        '''

        w = w - alpha * dw
        b = b - alpha * db

        if i % 100 == 0:
            costs.append(cost)
            print("the number of iteration: %i      the error: %f " % (i, cost))

    Y_prediction_train = predict(w, b, X)
    Y_prediction_test = predict(w, b, X_test)

    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    # plot the relationship between cost and iteration
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()


if __name__ == '__main__':
    main()





