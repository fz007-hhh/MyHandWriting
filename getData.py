import input_data # 调用input_data

print('Download and Extract MNIST dataset')
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print('type of "mnist" is %s' % (type(mnist)))
print('number of train data is %d' % (mnist.train.num_examples))
print('number of work data is %d' % (mnist.test.num_examples))