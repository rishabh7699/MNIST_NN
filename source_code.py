import tensorflow as tf
import idx2numpy as inp

# loding dataset
file1  = "train-images.idx3-ubyte"
ndarr1 = inp.convert_from_file(file1)

file2  = "train-labels.idx1-ubyte"
ndarr2 = inp.convert_from_file(file2)

file3  = "t10k-images.idx3-ubyte"
ndarr3 = inp.convert_from_file(file3)

file4  = "t10k-labels.idx1-ubyte"
ndarr4 = inp.convert_from_file(file4)


l     = 5                         # no. of layers
m     = 5000                      # train dataset size
t_m   = 1000                      # test dataset size
s     = [784,500,500,500,10]      # no. of nodes in each layer
steps = 30                        # no. of steps
bsize = 100                       # batch size
lambd = 0.01                      # learning rate


# function to print accuracy 
def print_accuracy(Input,Output,string,l):  
  res=s.run(tf.argmax(ans4,1),feed_dict={x: Input, prob: 1.0})
  c=0
  for i in range(l):
    if res[i]==Output[i]:
      c+=1
  print(string+" accuracy is:",str((c/l)*100))


# modifying input accoring to batch size, traning set size and test set size 
ndarr1=ndarr1[0:m]
ndarr2=ndarr2[0:m]
ndarr1=ndarr1.reshape(m,28*28)
ndarr3=ndarr3[0:t_m]
ndarr4=ndarr4[0:t_m]
ndarr3=ndarr3.reshape(t_m,28*28)

Ibatch=[]
Obatch=[]
for i in range(m//bsize):
  Ibatch.append(ndarr1[i*bsize:i*bsize+bsize])
  Obatch.append(ndarr2[i*bsize:i*bsize+bsize])


x=tf.placeholder(dtype=tf.float64)
y=tf.placeholder(dtype=tf.int64)
prob=tf.placeholder(tf.float64)


# initializing values for weight and bias with random numbers
w=[]
b=[]
w.append(tf.Variable( tf.random_normal(shape=[s[0],s[1]] ,dtype=tf.float64) ,dtype=tf.float64))
b.append(tf.Variable( tf.random_normal(shape=[s[1]]      ,dtype=tf.float64) ,dtype=tf.float64))

w.append(tf.Variable( tf.random_normal(shape=[s[1],s[2]] ,dtype=tf.float64) ,dtype=tf.float64))
b.append(tf.Variable( tf.random_normal(shape=[s[2]]      ,dtype=tf.float64) ,dtype=tf.float64))

w.append(tf.Variable( tf.random_normal(shape=[s[2],s[3]] ,dtype=tf.float64) ,dtype=tf.float64))
b.append(tf.Variable( tf.random_normal(shape=[s[3]]      ,dtype=tf.float64) ,dtype=tf.float64))

w.append(tf.Variable( tf.random_normal(shape=[s[3],s[4]] ,dtype=tf.float64) ,dtype=tf.float64))
b.append(tf.Variable( tf.random_normal(shape=[s[4]]      ,dtype=tf.float64) ,dtype=tf.float64))


# Model
ans1  = tf.nn.relu(tf.matmul(x,w[0])+b[0])
ans2_ = tf.nn.relu(tf.matmul(ans1,w[1])+b[1])
ans2  = tf.nn.dropout(ans2_,keep_prob=prob)
ans3  = tf.nn.relu(tf.matmul(ans2,w[2])+b[2])
ans4  = tf.matmul(ans3,w[3])+b[3]


cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=ans4))
optimizer=tf.train.AdamOptimizer(lambd)
train=optimizer.minimize(cost)


init=tf.global_variables_initializer()
s=tf.Session()
s.run(init)

# running optimizer steps no. of times
for i in range(steps):
  print("\nstep no:",str(i))
  
  for j in range(m//bsize):
    s.run(train,feed_dict={x: Ibatch[j],y: Obatch[j],prob: 0.5})
    
  print_accuracy(ndarr1,ndarr2,"Train",m)
  print_accuracy(ndarr3,ndarr4,"Test",t_m)
  
s.close()
