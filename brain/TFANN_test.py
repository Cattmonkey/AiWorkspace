# coding:utf-8
import tensorflow as tf
import GetMNIST_data as input_data
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
#matrix1 = tf.constant([[3., 3.]])
#print (matrix1)

## 创建另外一个常量 op, 产生一个 2x1 矩阵.
#matrix2 = tf.constant([[2.],[2.]])
#print (matrix1)
## 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
## 返回值 'product' 代表矩阵乘法的结果.
#product = tf.matmul(matrix1, matrix2)
#print (product)

#feed 使用一个 tensor 值临时替换一个操作的输出结果. 
#你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 
#方法结束, feed 就会消失. 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 
#标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
#input1 = tf.placeholder(tf.float32)
#input2 = tf.placeholder(tf.float32)
#output = tf.multiply(input1, input2)

#with tf.Session() as sess:
    #print (sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
    
#-----------------------------------------------------------------------

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
##更加方便的InteractiveSession类。通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，
##插入一些计算图，这些计算图是由某些操作(operations)构成的。
#sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
#W = tf.Variable (tf.zeros([784, 10]))
#d = tf.Variable(tf.zeros([10]))


#sess.run(tf.global_variables_initializer())

#y = x = tf.placeholder("float", shape=[None, None])

##训练模型
#y_ = tf.placeholder("float", [None,10])
##计算交叉熵  义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），
##然后尽量最小化这个指标。但是，这两种方式是相同的。
##一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
##TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用
##反向传播算法(backpropagation algorithm)
##来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
##然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
##train_step = tf.global_variables_initializer()
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

##sess = tf.Session()
##sess.run (init)

##开始训练模型
#for i in range(1000):
    #batch = mnist.train.next_batch(50)
    #train_step.run(feed_dict={x: batch[0], y_: batch[1]})

##评估我们的模型 
##tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
##这行代码会给我们一组布尔值。
##为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
##例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#--------------------------------------------------------------------------------------------------
#构建一个多层卷积网络

#为了创建这个模型，我们需要创建大量的权重和偏置项。
#这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
#由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，
#以避免神经元节点输出恒为0的问题（dead neurons）。为了不在建立模型的时候反复做初始化操作，
#我们定义两个函数用于初始化。

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积和池化
#TensorFlow在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，
#我们会一直使用vanilla版本。我们的卷积使用1步长（stride size），0边距（padding size）的模板，
#保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling。
#为了代码更简洁，我们把这部分抽象成一个函数。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#第一层卷积
#它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
#卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，
#接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
#最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积 
#为了构建一个更深的网络，我们会把几个类似的层堆叠起来。
#第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层 现
#在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，
#用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，
#加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，我们在输出层之前加入dropout。
#我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
#这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。

#输出层 
#最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#-----------------------------------------------------------------------