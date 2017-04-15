import tensorflow as tf

m1 = tf.constant([[3, 3]], dtype=tf.float32)
m2 = tf.constant([[2], [2]], dtype=tf.float32)
product = tf.matmul(m2, m1)

with tf.Session() as sess:
    res = sess.run(product)
    print(res)

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
print(sess.run(state))
for _ in range(5):
    sess.run(update)
    print(sess.run(state))
sess.close()

# feeds
input1 = tf.placeholder(tf.int8)
input2 = tf.placeholder(tf.int8)
output = tf.mul(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [1, 12], input2: [3, 4]}))
