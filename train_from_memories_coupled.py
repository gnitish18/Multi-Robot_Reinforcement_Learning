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
        self.drop = tf.keras.layers.Dropout(0.20)
        self.gauss = tf.keras.layers.GaussianNoise(stddev=0.2)
        #self.n1 = tf.keras.layers.BatchNormalization()
        #self.n2 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(48, activation='relu')
        self.d2 = tf.keras.layers.Dense(48*4, activation='relu')
        self.d3 = tf.keras.layers.Dense(48*8, activation='relu')
        self.d4 = tf.keras.layers.Dense(48*4, activation='relu')
        self.d5 = tf.keras.layers.Dense(48, activation='relu')
        
        # size 4, so that each teammate has action space of (up, down)
        # output here is Q value for each possible action for each teammate, which gets added together in loss function for total max q-value
        self.d6 = tf.keras.layers.Dense(4)
        
        ###############################################
        
    def call(self, x):
        x = self.gauss(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.d3(x)
        x = self.drop(x)
        x = self.d4(x)
        x = self.drop(x)
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

    #curr_action = tf.math.argmax(curr_output, 1)
#    curr_output = tf.round(curr_output)
#    target_output = tf.round(target_output)
    
    Q1 = tf.gather(curr_output[:, 0:2], tf.math.argmax(curr_output[:, 0:2], 1), axis=1) + tf.gather(curr_output[:, 2:4], tf.math.argmax(curr_output[:, 2:4], 1), axis=1)
    
    Q2 = tf.gather(target_output[:, 0:2], tf.math.argmax(target_output[:, 0:2], 1), axis=1) + tf.gather(target_output[:, 2:4], tf.math.argmax(target_output[:, 2:4], 1), axis=1)
    
    y = gamma*Q2 + reward[:,0]
    
    loss = tf.keras.losses.MSE(y, Q1)
    return loss
    
def train_nn(memories, curr_model, prev_model):
#################################################
### Tune these parameters for better training
    lr = 0.00000025
    #lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=100, decay_rate=0.5)
    epochs = 100
    batch_size = 50
  #################################################
#    lr = 0.00000025
#    epochs = 100 # best so far!
#    batch_size = 50
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    
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
            current_loss = loss(curr_model(state), action, reward, prev_model(next_state))
        
        # Only training current/main model, target model is reset elsewhere
        grad = tape.gradient(current_loss, curr_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, curr_model.trainable_variables))
        train_loss(current_loss)

    train_data = []
#    for i in range(memories[0].shape[0]):
#        state = memories[0][i,:]
#        action = memories[1][i,:]
#        reward = memories[2][i,:]
#        next_state = memories[3][i,:]
#        train_data.append(np.concatenate((state, action, reward, next_state)))
    
    # Process training data:
    data_size = 10000 # *needs argument
    #idx = int(np.floor(np.random.random()*(memories[0].shape[0] - data_size)))
    for i in range(data_size):
        idx = int(np.floor(np.random.random()*(memories[0].shape[0]))) # *Not sure what's going on here
            #if idx + i < memories[0].shape[0]:
        state = memories[0][idx,:]
        action = memories[1][idx,:]
        reward = memories[2][idx,:]
        next_state = memories[3][idx,:]
        train_data.append(np.concatenate((state, action, reward, next_state)))

    # could shuffle here. I'm unclear on randomizing each step or maintaining order
    # Randomizing data simply stirs up the problem a bit*
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
#        if epoch % 50 == 0:
#            prev_model = curr_model
    curr_model.save_weights('./trained_weights/foosPong_model_integrated')
    return curr_model
    
#    curr_model.summary()
#    #Saves model
#    curr_model.save_weights('./trained_weights/foosPong_model_v0')






