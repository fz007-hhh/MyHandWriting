# 随机取batch_size个训练样本
import numpy as np

class Method:
    def next_batch(self,train_data, train_target, batch_size):
        index = [ i for i in range(0,len(train_target)) ]
        np.random.shuffle(index)
        batch_data = []
        batch_target = []
        for i in range(0,batch_size):
            batch_data.append(train_data[index[i]])
            batch_target.append(train_target[index[i]])
        return batch_data, batch_target
