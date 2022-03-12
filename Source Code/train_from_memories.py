import pickle
import json
import tensorflow as tf
import numpy as np
import os

## Next steps to take
# 1. Create base memory from a bunch of episodes with my baseAI
# 2. Start training a NN with memories and baseAI as Q function (using Q-learning and rewards...)
# 3. Create more memories with new the NN weights loaded as Q function
# 4. Load weight and retrain with new memories
# 5. Repeat

### * Idea: crank up bounce speed and paddle speed and then train on PS/BS or vice-versa
###          - maybe it would learn to work with any PS/BS ratio....?
## prev_model = loadModel() or simpleModel

# NOTE: numbers need to be in native python 'float', NOT numpy.float32 (use .item() to convert)
def write2json(data,path,fname): # NOTE: ONLY takes lists as input, no ndarrays
    if not os.path.exists(path): # Create dir if doesn't already exist
        os.makedirs(path)
    with open(os.path.join(path,fname),'w') as output: # writing 'wb' here screws this mess up royally:
        # Raises "a bytes-like object is required, not 'str'", even when using '.tolist()' on component matrix
        json.dump(data,output)
        # pickle.dump(data,output)
        # output.write(json.dumps(data))

def loss(curr_output, action, reward, target_output,gamma):
    # gamma = 0.95

    Q1 = tf.gather(curr_output[:, 0:2], tf.math.argmax(curr_output[:, 0:2], 1), axis=1) + tf.gather(curr_output[:, 2:4], tf.math.argmax(curr_output[:, 2:4], 1), axis=1)
    
    Q2 = tf.gather(target_output[:, 0:2], tf.math.argmax(target_output[:, 0:2], 1), axis=1) + tf.gather(target_output[:, 2:4], tf.math.argmax(target_output[:, 2:4], 1), axis=1)
    
    y = gamma*Q2 + reward[:,0]
    
    loss = tf.keras.losses.MSE(y, Q1)
    return loss
    
def train_nn(lr, memories, curr_model, prev_model, gamma, epochs, batch_size, train_set_size, totalPaddles, noBalls, side, savedir):
    #################################################
    ### Tune these parameters for better training
    # lr = 0.0000025
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=100, decay_rate=0.5)
    # epochs = 15
    # batch_size = 10
    #################################################
    # lr = 0.00000025
    # epochs = 100 # best so far!
    # batch_size = 50
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    @tf.function
    def train(train_data,totalPaddles,noBalls):
        for tensor in train_data:
            train_step(tensor,totalPaddles,noBalls)
            
    @tf.function
    def train_step(tensor,totalPaddles,noBalls):
        trainPaddles = 2 # *Change if training more paddles
        dimstate = 2*totalPaddles + 4*noBalls # Computed dimension of state space based on how many things
        state = tensor[:, :dimstate]
        action = tensor[:, dimstate:(dimstate+trainPaddles)]
        reward = tensor[:, (dimstate+trainPaddles):(dimstate+trainPaddles+2)] # *Not sure what this 2 here is supposed to be
        next_state = tensor[:, (dimstate+2*trainPaddles):]
        
        with tf.GradientTape() as tape:
            current_loss = loss(curr_model(state), action, reward, prev_model(next_state),gamma)

        grad = tape.gradient(current_loss, curr_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, curr_model.trainable_variables))
        train_loss(current_loss)

    train_data = []
    # for i in range(memories[0].shape[0]):
    #     state = memories[0][i,:]
    #     action = memories[1][i,:]
    #     reward = memories[2][i,:]
    #     next_state = memories[3][i,:]
    #     train_data.append(np.concatenate((state, action, reward, next_state)))

    data_size = train_set_size
    # idx = int(np.floor(np.random.random()*(memories[0].shape[0] - data_size)))
    for i in range(data_size):
        idx = int(np.floor(np.random.random()*(memories[0].shape[0])))
            #if idx + i < memories[0].shape[0]:
        state = memories[0][idx,:]
        action = memories[1][idx,:]
        reward = memories[2][idx,:]
        next_state = memories[3][idx,:]
        train_data.append(np.concatenate((state, action, reward, next_state)))
        
    # could shuffle here. Unclear on randomizing each step or maintaining order
    train_data_tf = tf.data.Dataset.from_tensor_slices(train_data).shuffle(50000).batch(batch_size)
    # train_data_tf = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
    
    train_loss_save = []
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train(train_data_tf,totalPaddles,noBalls)
        # print("works")
        template = '\nEpoch {}, Loss: {}\n'
        print(template.format(epoch + 1, train_loss.result()))
        train_loss_save.append(train_loss.result().numpy().item()) # Convert to appropriate format
        # updates target network
        # if epoch % 50 == 0:
        #     prev_model = curr_model
    
    # Define output dir
    path=savedir
    
    # Save loss and other metrics to .json files -- note, this fcn will create the path if it doesn't exist        
    write2json(train_loss_save,path,fname='train_loss.json')
    print("Saving training loss...")
    
    # Save tf weights
    if side == 0:
        curr_model.save_weights(path+'foosPong_model_right')
    else:
        curr_model.save_weights(path+'foosPong_model_left')

    return curr_model
        
    # curr_model.summary()
    # # Saves model
    # curr_model.save_weights('./trained_weights/foosPong_model_v0')