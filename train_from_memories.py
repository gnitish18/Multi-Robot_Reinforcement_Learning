import pickle
import tensorflow as tf
import numpy as np

## Next steps to take
# 1. Create base memory from a bunch of episodes with my baseAI
# 2. Start training a NN with memories and baseAI as Q function (using Q-learning and rewards...)
# 3. Create more memories with new the NN weights loaded as Q function
# 4. Load weight and retrain with new memories
# 5. Repeat

### * Idea: crank up bounce speed and paddle speed and then train on PS/BS or vice-versa
###          - maybe it would learn to work with any PS/BS ratio....?


## prev_model = loadModel() or simpleModel

class foosPong_model(tf.keras.Model):
    def __init__(self):
        super(foosPong_model, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.15)
        
        self.n1 = tf.keras.layers.BatchNormalization()
        self.n2 = tf.keras.layers.BatchNormalization()
        #self.n3 = tf.keras.layers.BatchNormalization()
        #self.n4 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(100)
#        self.d2 = tf.keras.layers.Dense(200, activation='relu')
#        self.d3 = tf.keras.layers.Dense(200)
        self.d4 = tf.keras.layers.Dense(100, activation='relu')
        self.d5 = tf.keras.layers.Dense(50)
        self.d6 = tf.keras.layers.Dense(2, activation='softmax')
        
        ###############################################
        
    def call(self, x):
        x = self.d1(x)
        #x = self.drop(x)
        
#        x = self.d2(x)
#        x = self.n1(x)
#
#        x = self.d3(x)
#        #x = self.drop(x)
        
        x = self.d4(x)
        #x = self.n2(x)
        
        x = self.d5(x)
        return self.d6(x)

#def pong_ai(tensor):
#
#    if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
#        return 1
#    else:
#        return 0

def loss(curr_output, action, reward):
    #curr_action = tf.math.argmax(curr_output, 1)
    
    Q1 = curr_output[:,0] + curr_output[:,1]
    Q2 = 0.99*(np.random.random() + np.random.random()) + reward[:,0]
    
    loss = tf.keras.losses.MSE(Q1, Q2)
    return loss
    
def train_nn(memories, hasPrevModel=False):
    lr = 0.00001
    epochs = 100
    batch_size = 1000
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    curr_model = foosPong_model()
    
    if hasPrevModel:
        prev_model = foosPong_model()
        prev_model.load_weights('foosPong_model_weights_0')
    
    
    @tf.function
    def train(train_data):
        for tensor in train_data:
            train_step(tensor)


    @tf.function
    def train_step(tensor):
        state = tensor[:, :24]
        action = tensor[:, 24:26]
        reward = tensor[:, 26:28]
        next_state = tensor[:, 28:]
        
        with tf.GradientTape() as tape:
            current_loss = loss(curr_model(state), action, reward)

        grad = tape.gradient(current_loss, curr_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, curr_model.trainable_variables))
        train_loss(current_loss)

    train_data = []
    data_size = 100000
    for i in range(data_size):
        idx = int(np.floor(np.random.random()*memories[0].shape[0]))
        
        state = memories[0][idx,:]
        action = memories[1][idx,:]
        reward = memories[2][idx,:]
        next_state = memories[3][idx,:]
        
        train_data.append(np.concatenate((state, action, reward, next_state)))
    
    
    train_data_tf = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
    
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train(train_data_tf)
        #print("works")
        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))

    
    
###########################################################################
###########################################################################
#####################           Training begins         ###################
###########################################################################
###########################################################################
    
with open("memory_states.txt", "rb") as fp:
    memory_states = np.asarray(pickle.load(fp), dtype='float32')
    
with open("memory_actions.txt", "rb") as fp:
    memory_actions = np.asarray(pickle.load(fp), dtype='float32')
    
with open("memory_rewards.txt", "rb") as fp:
    memory_rewards = np.asarray(pickle.load(fp), dtype='float32')
    
with open("memory_next_states.txt", "rb") as fp:
    memory_next_states = np.asarray(pickle.load(fp), dtype='float32')

memories = [memory_states, memory_actions, memory_rewards, memory_next_states]

train_nn(memories, hasPrevModel=False)





