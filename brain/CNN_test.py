# coding:utf-8
import tensorflow as tf
##tensorflow卷积定义 relu(W*X+B) W 矩阵 * X矩阵 + B矩阵 = 
##W 权重variable变量 * X  (placeholder占位符外部输入)variable变量 + B偏重变量
##因为深度学习 会自动 不断地计算loss损失 BP 来调整 w b 所以 w b初始化 可以随便 全部都是0 都行 
##对于 X来说其实我们知道 就是 我们图像数据 Y 是图像的标签，但是Y需要转为数学可以计算的值，所以采用one-hot 数组记录 标签的索引就行，比如 xx1 xx2 xx3  相应的y1=[1,0,0] y2=[0 1 0] y3=[0 0 1]
##那么 X 图像的像素 通过外部输入 placeholder占位符  Y值 外部输入 通过 placeholder占位符 

##Sigmoid: f(x)=1/(1+e^-x)。神经元的非线性作用函数:它能够把输入的连续实值“压缩”到0和1之间。 
##ReLU:    f(x)=max(0,x) 输入信号<0时，输出都是0，>0 的情况下，输出等于输入 

##下面是一个卷积层 定义 relu(wx+b) 是tensorflow来表示relu(wx+b)的公式 
##其中要注意参数 strides 是卷积滑动的步长 你可以配置更多的系

#w = 10
#h = 10

#def conv2d (x, w, b):
    #return (tf.nn.relu (tf.nn.bias_add (tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding='SAME'), b)))

#x = tf.placeholder (tf.float32, [None, w*h])#w*h 因为批次训练 数据可以任意所以第一个是None ，第二个图像是个w * h的图像 ，可以展开得到 w * h 的数组
#y = tf.placeholder(tf.float32, [None, ysize])#y的数目个数 比如3个类 就是3

##那么 一个x 需 要 *  W = y   计入batch为50 y为10  那么[50,224*224] * W= [50 ,10] 
##那么W 需要是一个[224*224，10] 的矩阵 ，也就是说 至少224*224*10*50 个连接

##X [None ,w*h] 对于每一个 w*h 是一个矩阵, 每一层的w 也是一个矩阵 每一层的 b也是一个矩阵
##每一层的输出y1 也是一个矩阵 y=[w*h]*w+b 为了减少系数，使用卷积，把它转换成MXN的值 
##这里就是跟全连接层的不同，使用了卷积转换成了一个MXN的卷积特征 而全连接层就是 y=wx+b 

##因为卷积层的w是需要 与 w*h 提取的MXK来做矩阵相乘 所以 他是跟卷积核相关 以及 输入 输出 相关，对于每一张图像

#wc=tf.Variable(tf.random_normal([3, 3, 1, 64]))
##3 3 分别为3x3大小的卷积核 1位输入数目 因为是第一层所以是1 输出我们配置的64 
##如果下一次卷积wc2=[5,5,64,256] //5x5 是我们配置的卷积核大小
##第三位表示输入数目 我们通过上面知道 上面的输出 也就是下一层的输入

##这样我们知道了 wc*(卷积得到的x) + b =y1下一层输入，
##我们也能知道 一下层输入 mxnxo 分别为输出O个卷积核特征MXN 我们也能知道b 大小 那个肯定跟O一样，
##比如上面是64输出，所以

#b1=tf.Variable(tf.random_normal([64]))
##同样可以知道b2=[256]

## 下面我们讲一讲池化层pool 池化层 不会减少输出，只会把MXNXO ，
##把卷积提取的特征 做进一步卷积 取MAX AVG MIN等 用局部代替整理进一步缩小特征值大小

#def max_pool_kxk(x):
    #return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        #strides=[1, k, k, 1], padding='SAME')
                        
##定义一个简单的CNN
#c1 = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
#m1 = tf.nn.max_pool(l1a, ksize=[1, k, k, 1],strides=[1, k, k, 1], padding='SAME')
#d1 = tf.nn.dropout(l1, p_keep_conv)

#c2 = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
#m2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#d2= tf.nn.dropout(l2, p_keep_conv)

#c3 = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
#m3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#d3 = tf.nn.dropout(tf.reshape(l3, [-1, w4.get_shape().as_list()[0]]), p_keep_conv)
##d3表示倒数第二层的输出 也就是倒数第一层一层的输入x 我们代入 y=wx+b 也就是w*x=w4*d3+b 
#y = tf.nn.bias_add(tf.matmul(d3, w4),b4)#matmul 顾名思义mat矩阵mul相乘 matmul w*x


import GetMNIST_data
mnist = GetMNIST_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 网络参数
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf 数据输入Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# 创建模型
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # 形状重定义
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # 层1Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # 层32onvolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # 全连接层Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 创建 model
pred = conv_net(x, weights, biases, keep_prob)

# 定义损失 和  优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估模型 model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.initialize_all_variables()

# 运行图
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 使用 batch data 抓训练数据
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # 计算 batch 准确性
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # 计算 batch 损失
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("优化 Finished!")
    #  256 mnist 测试集准确度
    print ("测试准确度:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
