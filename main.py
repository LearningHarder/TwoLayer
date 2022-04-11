import numpy as np
import os
import gzip
import json
import matplotlib.pyplot as plt
import random
from neural_net import TwoLayerNet

from builtins import range
from past.builtins import xrange

from math import sqrt, ceil


# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

def load_data(data_folder):

  files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]

  paths = []
  for fname in files:
    paths.append(os.path.join(data_folder,fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8).astype("int")

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28).astype("float")

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8).astype("int")

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28).astype("float")

  return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test)= load_data('MNIST_data/')
print('-----数据集初始维度-----')
print("train_images shape:",X_train.shape)
print("train_labels shape:",y_train.shape)
print("test_images shape:",X_test.shape)
print("test_labels shape:",y_test.shape)
num_training,num_validation,num_test = 54000,6000,10000
# num_training,num_validation,num_test,num_dev = 10000,6000,10000,1000

# subsample the data
mask = list(range(num_training, num_training + num_validation))
X_val = X_train[mask]
y_val = y_train[mask]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]


# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


 # Normalize the data: subtract the mean image
mean_image = np.mean(X_train, axis = 0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image


# # add bias dimension and transform into columns
# X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
# X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
# X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print('-----数据集input维度------')
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


labels = set(y_train)
print("labels:",labels)

input_size = 28 * 28
hidden_size = 100
num_classes = 10
random.seed(0)

best_net = None # store the best model into this 


'''
#### 网格搜索
results = {}
best_val_acc = 0
best_net = None


# 4 * 3 * 3 = 36
hidden_size = [50, 100, 150, 200]
learning_rates = [0.001,0.005,0.01]
regularization_strengths = [0.05, 0.1, 0.15]
# hidden_size = [50]
# learning_rates = [0.001,0.005]
# regularization_strengths = [0.05]

f = open('log.txt','w')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:            
            net = TwoLayerNet(input_size, hs, num_classes)
            # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=2000, batch_size=128,
            learning_rate=lr, learning_rate_decay=0.95,
            reg= reg, verbose=False)
            # net.dropout(p=0.1)
            train_acc = (net.predict(X_train) == y_train).mean()
            val_acc = (net.predict(X_val) == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net  
                best_hs = hs
                best_net.save_model(model= 'bestmodelparams.npz')     
            results[(hs,lr,reg)] = val_acc
            print ('hs %d lr %e reg %e train accuracy: %f val accuracy: %f' % (hs, lr, reg, train_acc, val_acc))
            f.write('hs %d lr %e reg %e train accuracy: %f val accuracy: %f \n' % (hs, lr, reg, train_acc, val_acc))
f.close()
# with open("result.json", "w", encoding='utf-8') as f:
#     # json.dump(dict_, f)  # 写为一行
#     json.dump(results, f)  # 写为多行
'''

#### 随机搜索
results = {}
best_val_acc = 0
best_net = None


# 4 * 3 * 3 = 36
hidden_size = [50, 100, 150, 200]
learning_rates = [0.001,0.005,0.01]
regularization_strengths = [0.05, 0.1, 0.15]
# hidden_size = [50]
# learning_rates = [0.001,0.005]
# regularization_strengths = [0.05]

best_stats = None
f = open('log_randomsearch.txt','w')
for i in range(100):
    hs = random.randint(10,40) * 10
    lr = random.uniform(0.001,0.01)
    reg = random.uniform(0.05,0.15)          
    net = TwoLayerNet(input_size, hs, num_classes)
    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
    num_iters=1500, batch_size=100,
    learning_rate=lr, learning_rate_decay=0.95,
    reg= reg, verbose=False)
    # net.dropout(p=0.1)
    train_acc = (net.predict(X_train) == y_train).mean()
    val_acc = (net.predict(X_val) == y_val).mean()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_net = net 
        best_stats = stats 
        best_hs = hs
        best_net.save_model(model= 'bestmodelparams_randomsearch.npz')     
    results[(hs,lr,reg)] = val_acc
    print ('hs %d lr %e reg %e train accuracy: %f val accuracy: %f' % (hs, lr, reg, train_acc, val_acc))
    f.write('hs %d lr %e reg %e train accuracy: %f val accuracy: %f \n' % (hs, lr, reg, train_acc, val_acc))
f.close()



# Print out results.
for hs,lr, reg in sorted(results):
    val_acc = results[(hs, lr, reg)]
    print ('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg,  val_acc))
    
print ('best validation accuracy achieved during cross-validation: %f' % best_val_acc)


## 对test进行预测
net = TwoLayerNet(input_size, best_hs, num_classes)
# net.load_model()
net.load_model('bestmodelparams_randomsearch.npz')
test_acc = (net.predict(X_test) == y_test).mean()
print ('test accuracy achieved during cross-validation: %f' % test_acc)



# net = TwoLayerNet(input_size, hidden_size, num_classes)

# # Train the network
# stats = net.train(X_train, y_train, X_val, y_val,
#             num_iters=3000, batch_size=128,
#             learning_rate=1e-4, learning_rate_decay=0.95,
#             reg=0.05, verbose=True)

# # Predict on the validation set
# val_acc = (net.predict(X_val) == y_val).mean()
# print('Validation accuracy: ', val_acc)
# test_acc = (net.predict(X_test) == y_test).mean()
# print('Test accuracy: ', test_acc)

# Plot the loss function and train / validation accuracies
plt.figure(figsize=(16,6),dpi=300)
plt.subplot(1, 2, 1)
plt.plot(best_stats['train_loss_history'])
plt.title('Train')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(best_stats['val_loss_history'])
plt.title('Val')
plt.xlabel('Iteration')
plt.ylabel('Loss')
# plt.show()

plt.savefig('images/loss.jpg')


# 绘制acc曲线
plt.figure(figsize=(16,6),dpi=300)
plt.subplot(1, 2, 1)
plt.plot(best_stats['train_acc_history'])
plt.title('Train')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(best_stats['val_acc_history'])
plt.title('Val')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
# plt.show()

plt.savefig('images/acc.jpg')

### 可视化参数W1

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid



def show_net_weights(W1):
#     W1 = net.params['W1']
    W1 = W1.reshape(28, 28, 1, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig('images/w1_vis.jpg')

params = {}
p = np.load('bestmodelparams_randomsearch.npz')
params['W1'] = p['w1']
params['W2'] = p['w2']
params['b1'] = p['b1']
params['b2'] = p['b2']
show_net_weights(params['W1'])

## 可视化W2
plt.imshow(params['W2'].transpose(1,0))
plt.gca().axis('off')
plt.savefig('images/w2_vis.jpg')




