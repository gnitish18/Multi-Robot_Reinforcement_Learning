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
        #self.drop = tf.keras.layers.Dropout(0.10)
        
        self.n1 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(48, activation='relu')
        self.d2 = tf.keras.layers.Dense(96, activation='relu')
        self.d3 = tf.keras.layers.Dense(192, activation='relu')
        self.d4 = tf.keras.layers.Dense(96, activation='relu')
        self.d5 = tf.keras.layers.Dense(48, activation='relu')
        
        # size 4, so that each teammate has action space of (up, down)
        # output here is Q value for each possible action for each teammate, which gets added together in loss function for total max q-value
        self.d6 = tf.keras.layers.Dense(4, activation='relu')
        
        ###############################################
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.n1(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.d6(x)

#def pong_ai(tensor):
#
#    if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
#        return 1
#    else:
#        return 0

def loss(curr_output, action, reward, target_output):
    gamma = 0.95

    # model_output is shape (1,4),
    # which is [paddle1 action1, paddle1 action2, paddle2 action1, paddle2 action2]
    
    # each position is the Q value for that paddle taking that action
    # Thus, here we take the position of the max Q-value for each paddle
    
    # the target_model gets reset to equal curr_model every 100 epochs or whatever you want
    
    Q1 = tf.gather(curr_output[:, 0:2], tf.math.argmax(curr_output[:, 0:2], 1), axis=1) + tf.gather(curr_output[:, 2:4], tf.math.argmax(curr_output[:, 2:4], 1), axis=1)
    
    Q2 = tf.gather(target_output[:, 0:2], tf.math.argmax(target_output[:, 0:2], 1), axis=1) + tf.gather(target_output[:, 2:4], tf.math.argmax(target_output[:, 2:4], 1), axis=1)
    
    # The reward postions are the same. By only adding the 0-index I'm considering that number as a joint reward. This could certainly change
    y = gamma*Q2 + reward[:,0]
    
    loss = tf.keras.losses.MSE(Q1, y)
    return loss
    
def train_nn(memories, hasPrevModel=False):
    #################################################
    ### Tune these parameters for better training
    lr = 0.001
    epochs = 1000
    batch_size = 10000
    #################################################  
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    curr_model = foosPong_model()
    prev_model = foosPong_model()
    
    if hasPrevModel:
        curr_model.load_weights('./trained_weights/foosPong_model_v0')
        prev_model.load_weights('./trained_weights/foosPong_model_v0')
    
    @tf.function
    def train(train_data):
        steps = 0
        for tensor in train_data:
            # #updates the target model after so many batches
            # if steps % 1000 == 0:
            #     prev_model = curr_model
                
            train_step(tensor)
            steps = steps + 1
    
    @tf.function
    def train_step(tensor):
        state = tensor[:, :24]
        action = tensor[:, 24:26]
        reward = tensor[:, 26:28]
        next_state = tensor[:, 28:]
        
        with tf.GradientTape() as tape:
            current_loss = loss(curr_model(state), action, reward, prev_model(next_state))

        grad = tape.gradient(current_loss, curr_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, curr_model.trainable_variables))
        train_loss(current_loss)
    
    train_data = []
    #data_size = 100000
    for i in range(memories[0].shape[0]):
        #idx = int(np.floor(np.random.random()*memories[0].shape[0]))
        state = memories[0][i,:]
        action = memories[1][i,:]
        reward = memories[2][i,:]
        next_state = memories[3][i,:]
        
        train_data.append(np.concatenate((state, action, reward, next_state)))
    
    # could shuffle here. I'm unclear on randomizing each step or maintaining order
    train_data_tf = tf.data.Dataset.from_tensor_slices(train_data).shuffle(50000).batch(batch_size)
    #train_data_tf = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
    
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train(train_data_tf)
        #print("works")
        template = '\nEpoch {}, Loss: {}\n'
        print(template.format(epoch + 1, train_loss.result()))
        
        #updates target network
        if epoch % 100 == 0:
            prev_model = curr_model

    curr_model.summary()
    #Saves model
    curr_model.save_weights('./trained_weights/foosPong_model_v0')
    
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

###########################################################################
# This function starts it all. Set hasPrevModel=False for first trainging attempt

train_nn(memories, hasPrevModel=False)
