
# coding: utf-8

# In[18]:


import tensorflow as tf
import carga_entrenamiento as dataset
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# Resetear el grafo de computación
tf.reset_default_graph()
# Declarar sesión
sess = tf.Session()
# Placeholders para imágenes y clases de entrenamiento
x = tf.placeholder("float", shape=[None, 1024])
y_ = tf.placeholder("float", shape=[None, 36])
# Inicio de la declaracion de la arquitectura de red
with tf.name_scope("Reshaping_data") as scope:
    x_image = tf.reshape(x, [-1,32,32,1])

with tf.name_scope("Conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 64],"Conv_Layer_1")
    b_conv1 = bias_variable([64],"Bias_Conv_Layer_1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
with tf.name_scope("Conv2") as scope:
    W_conv2 = weight_variable([3, 3, 64, 64],"Conv_Layer_2")
    b_conv2 = bias_variable([64],"Bias_Conv_Layer_2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("Conv3") as scope:
    W_conv3 = weight_variable([3, 3, 64, 64],"Conv_Layer_3")
    b_conv3 = bias_variable([64],"Bias_Conv_Layer_3")
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)


with tf.name_scope("Fully_Connected1") as scope:
    W_fc1 = weight_variable([4 * 4 * 64, 1024],"Fully_Connected_layer_1")
    b_fc1 = bias_variable([1024],"Bias_Fully_Connected1")
    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
with tf.name_scope("Fully_Connected2") as scope:
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 36],"Fully_Connected_layer_2")
    b_fc2 = bias_variable([36],"Bias_Fully_Connected2")

with tf.name_scope("Final_Softmax") as scope:
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
with tf.name_scope("Entropy") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope("evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
saver = tf.train.Saver()
# Inicialización del grafo
sess.run(tf.global_variables_initializer())
indice = 0
batch_size = 300
total_epoch = 4500
resultados_accuracy_training = []
resultados_accuracy_validation = []
resultados_loss_train = []
# Recuperar red previamente entrenada (si existe)
ckpt = tf.train.get_checkpoint_state("c:/Users/KevinPercy/Downloads/CNN_log/")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print ("no se ha encontrado checkpoint")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.cargar_dataset() 
    # Centrado de datos de los subconjuntos
    X_train = X_train - np.mean(X_train, axis=0)
    X_val = X_val - np.mean(X_val, axis=0)
    X_test = X_test - np.mean(X_test, axis=0)
    # Bucle de iteraciones de entrenamiento
    for i in range(total_epoch):
        # Carga del batch de imágenes
        batch_x = X_train[indice:indice + batch_size]
        batch_y = y_train[indice:indice + batch_size]
        # Actualizamos el índice
        indice = indice + batch_size + 1
        if indice > X_train.shape[0]:
            indice = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)
        if i%10 == 0:
            results_train = sess.run([accuracy,cross_entropy],feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
            train_validation = sess.run(accuracy,feed_dict={x:X_val, y_: y_val,keep_prob: 1.0})
            train_accuracy = results_train[0]
            train_loss = results_train[1]
            resultados_accuracy_training.append(train_accuracy)
            resultados_accuracy_validation.append(train_validation)
            resultados_loss_train.append(train_loss)
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, validation accuracy %g"%(i, train_validation))
            print("step %d, loss %g"%(i, train_loss))
            # Guardar el modelo en cada iteración del entrenamiento
        saver.save(sess, 'CNN_log/model.ckpt', global_step=i+1)
        sess.run(train_step,feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

    print ("FINALIZADO training")
    # Visualizar precisión y error para subconjuntos de entrenamiento y validación
    eje_x = np.arange(total_epoch/10)
    array_training = np.asanyarray(resultados_accuracy_training)
    array_validation = np.asanyarray(resultados_accuracy_validation)
    array_loss_train = np.asanyarray(resultados_loss_train)
    plt.figure(1)
    linea_train, = plt.plot(eje_x,array_training[eje_x],label="train",linewidth=2)
    linea_test, = plt.plot(eje_x,array_validation[eje_x],label="validation",linewidth=2)
    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left', ncol=1)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    plt.figure(2)
    linea_loss, = plt.plot(eje_x,array_loss_train[eje_x],label="loss",linewidth=2)
    plt.legend(bbox_to_anchor=(1,1.02), loc='upper left', ncol=1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    # Calcular precisión para el subconjunto de test
    test_accuracy = sess.run( accuracy, feed_dict={x:X_test, y_: y_test, keep_prob: 1.0})
    print("test accuracy %g"% test_accuracy)
        
def reconocer_matricula(letras_matricula):
    matricula = ""
    clases =["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N"
    ,"O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    letras_matricula = np.matrix(letras_matricula)
    classification = sess.run(y_conv, feed_dict={x:letras_matricula,keep_prob:1.0})
    for p in range(classification.shape[0]):
        pred = sess.run(tf.argmax(classification[p,:], 0))
        matricula = matricula + clases[int(pred)]
    return matricula 

