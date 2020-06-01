
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib import rnn
import numpy as np
import tensorflow.contrib.slim as slim
from compute_mcc import *
import os
import math
import h5py
from hilbert import hilbertCurve
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
log_device_placement = True


lr = 0.00003
training_iters = 500
batch_size = 16
display_step = 10
nb_nontamp_img=0
nb_tamp_img=16
nbFilter=32



n_input = 240
n_steps = 64 
nBlock=int(math.sqrt(n_steps))
n_hidden = 64
nStride=int(math.sqrt(n_hidden))
imSize=256
n_classes = 2

input_layer = tf.placeholder("float", [None, imSize,imSize,3])
y= tf.placeholder("float", [2,None, imSize,imSize])
freqFeat=tf.placeholder("float", [None, 64,240])
ratio=15.0 
units_between_stride = 2
upsample_factor=16
n_classes=2
beta=.01
outSize=16
seq = np.linspace(0,63,64).astype(int)
order3 = hilbertCurve(3)
order3 = np.reshape(order3,(64))
hilbert_ind = np.lexsort((seq,order3))
actual_ind=np.lexsort((seq,hilbert_ind))

weights = {
    'out': tf.Variable(tf.random_normal([64,64,nbFilter]))
}
biases = {
    'out': tf.Variable(tf.random_normal([nbFilter]))
}




with tf.device('/gpu:1'):

    def conv_mask_gt(z): 
        
        class_labels_tensor = (z==1)  
        background_labels_tensor = (z==0)
        bit_mask_class = np.float32(class_labels_tensor)
        bit_mask_background = np.float32(background_labels_tensor)
        combined_mask=[]
        combined_mask.append(bit_mask_background)
        combined_mask.append(bit_mask_class)
        return combined_mask

    def get_kernel_size(factor):
        return 2 * factor - factor % 2

    def upsample_filt(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)

    def bilinear_upsample_weights(factor, number_of_classes):   
        filter_size = get_kernel_size(factor)

        weights = np.zeros((filter_size,filter_size,number_of_classes,number_of_classes), dtype=np.float32)    
        upsample_kernel = upsample_filt(filter_size)    
        for i in range(number_of_classes):        
            weights[:, :, i, i] = upsample_kernel    
        return weights


    def resUnit(input_layer,i,nbF):
      with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,nbF,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,nbF,[3,3],activation_fn=None)	
        output = input_layer + part6
        return output

    def segNet(input_layer,bSize,freqFeat,weights,biases):
        layer1 = slim.conv2d(input_layer,nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))	
        layer1 =resUnit(layer1,1,nbFilter)
        layer1 = tf.nn.relu(layer1)
        layer2=slim.max_pool2d(layer1, [2, 2], scope='pool_'+str(1))		
        layer2 = slim.conv2d(layer2,2*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(1))		
        layer2 =resUnit(layer2,2,2*nbFilter)
        layer2 = tf.nn.relu(layer2)
        layer3=slim.max_pool2d(layer2, [2, 2], scope='pool_'+str(2))
        layer3 = slim.conv2d(layer3,4*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(2))		
        layer3 =resUnit(layer3,3,4*nbFilter)
        layer3 = tf.nn.relu(layer3)
        layer4=slim.max_pool2d(layer3, [2, 2], scope='pool_'+str(3))
        layer4 = slim.conv2d(layer4,8*nbFilter,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(3))		
        layer4 =resUnit(layer4,4,8*nbFilter)
        layer4 = tf.nn.relu(layer4)		
        layer4=slim.max_pool2d(layer4, [2, 2], scope='pool_'+str(4))
       
        patches=tf.transpose(freqFeat,[1,0,2])
        patches=tf.gather(patches,hilbert_ind)
        patches=tf.transpose(patches,[1,0,2])         
        
        xCell=tf.unstack(patches, n_steps, 1)
       
        stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden),output_keep_prob=0.9) for _ in range(2)] )
        out, state = rnn.static_rnn(stacked_lstm_cell, xCell, dtype=tf.float32)
        
        out=tf.gather(out,actual_ind)
        lstm_out=tf.matmul(out,weights['out'])+biases['out']
        lstm_out=tf.transpose(lstm_out,[1,0,2])

        lstm_out=tf.reshape(lstm_out,[bSize,8,8,nbFilter])

        lstm_out=slim.batch_norm(lstm_out,activation_fn=None)
        lstm_out=tf.nn.relu(lstm_out)

        temp=tf.random_normal([bSize,outSize,outSize,nbFilter])
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(2, nbFilter)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        lstm_out = tf.nn.conv2d_transpose(lstm_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 2, 2, 1])

        top = slim.conv2d(layer4,nbFilter,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
        top = tf.nn.relu(top)
       
        joint_out=tf.concat([top,lstm_out],3)		
       
        temp=tf.random_normal([bSize,outSize*4,outSize*4,2*nbFilter])       //16*4 16*4 = (64,64)
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(4, 2*nbFilter)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_layer4 = tf.nn.conv2d_transpose(joint_out, upsample_filter_tensor,output_shape=uShape1,strides=[1, 4, 4, 1]) 	
       	
        upsampled_layer4 = slim.conv2d(upsampled_layer4,2,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_'+str(4))
        upsampled_layer4=slim.batch_norm(upsampled_layer4,activation_fn=None)
        upsampled_layer4=tf.nn.relu(upsampled_layer4)
        
        temp=tf.random_normal([bSize,outSize*16,outSize*16,2])              //16*16 16*16 = (256*256)
        uShape1=tf.shape(temp)
        upsample_filter_np = bilinear_upsample_weights(4,2)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_layer5 = tf.nn.conv2d_transpose(upsampled_layer4, upsample_filter_tensor,output_shape=uShape1,strides=[1, 4, 4, 1]) 
        
        return upsampled_layer5


    y1=tf.transpose(y,[1,2,3,0])
    upsampled_logits=segNet(input_layer,batch_size,freqFeat,weights,biases)


    flat_pred=tf.reshape(upsampled_logits,(-1,n_classes))
    flat_y=tf.reshape(y1,(-1,n_classes))

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(flat_y,flat_pred, 1.0))

    trainer = tf.train.AdamOptimizer(learning_rate=lr)
    update = trainer.minimize(loss)
   
    probabilities=tf.nn.softmax(flat_pred)
    correct_pred=tf.equal(tf.argmax(probabilities,1),tf.argmax(flat_y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    y_actual=tf.argmax(flat_y,1)
    y_pred=tf.argmax(flat_pred,1)

    mask_actual= tf.argmax(y1,3)
    mask_pred=tf.argmax(upsampled_logits,3)


init = tf.initialize_all_variables()
saver = tf.train.Saver()

config=tf.ConfigProto()
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    sess.run(init) 
    saver.restore(sess,'model/final_model_nist.ckpt')
    print('session starting .................!!!!' )

    feat1=h5py.File('train_data/train_data_feat.hdf5','r')
    freq1=np.array(feat1["feat"])
    feat1.close()
    mx=127.0
    hdf5_file=h5py.File('train_data/spliced_copymove_nist_imgs.hdf5','r')
    Img=np.array(hdf5_file["train_img"])
   
    Lab=np.array(hdf5_file["train_mask"])

    subtract_mean = True
    step = 1
    hdf5_file.close()
    dx=Img[-batch_size:]
    dx=np.multiply(dx,1.0/mx)
    dx1=freq1[-batch_size:]
    dy=Lab[-batch_size:]
    dy=conv_mask_gt(dy)
    print('epoch 1 sarted......')
    epoch_iter=0
    iter_nontamp=0;iter_tamp=0;iter_nist=0;iter_nc17=0;iter_nc16=0
    
    epoch_iter_tamp=int(nb_tamp_img/6)
    bTamp=6

    best_acc=np.float32(0.45)
    best_prec=np.float32(0.2)
    best_acc1=np.float32(0.45)
    best_prec1=np.float32(0.15)
    
    batch_x=np.zeros((batch_size,imSize,imSize,3))  
    batch_y=np.zeros((batch_size,imSize,imSize))    
    batch_x1=np.zeros((batch_size,64,240))          

    while step * batch_size < training_iters:
        if (iter_tamp % epoch_iter_tamp)==0:
            print("data loading for synthesized images ...")
            iter_tamp=0
            in_size=nb_tamp_img
            arr_ind=np.arange(in_size)      
            np.random.shuffle(arr_ind)      
            arr_ind=arr_ind+nb_nontamp_img  
            
            im2 = Img[arr_ind, ...]               
            Y2 = Lab[arr_ind, ...]          
            fr2=freq1[arr_ind, ...]         
            
            epoch_iter+=1
            print("epoch finished..starting next epoch..>>>")

            im2=im2[:(int)(np.shape(arr_ind)[0]/bTamp)*bTamp,...]   
            Y2=Y2[:(int)(np.shape(arr_ind)[0]/bTamp)*bTamp,...]     
            fr2=fr2[:(int)(np.shape(arr_ind)[0]/bTamp)*bTamp,...]
        batch_x[6:12,...]=np.float32(im2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,nb_tamp_img),...])
        batch_y[6:12,...]=Y2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,nb_tamp_img),...]
        batch_x1[6:12,...]= fr2[(iter_tamp*bTamp):min((iter_tamp+1)*bTamp,nb_tamp_img),...]
        iter_nontamp+=1;iter_tamp+=1;iter_nist+=1; iter_nc17+=1;iter_nc16+=1

        rev_batch_y=np.array(conv_mask_gt(batch_y))	
        if np.shape(batch_x)[0]!= batch_size:
            continue
        batch_x=np.multiply(batch_x,1.0/mx)
        sess.run(update, feed_dict={input_layer: batch_x, y: rev_batch_y, freqFeat: batch_x1})
        
        if step % display_step == 0:
            TP = 0; FP = 0;TN = 0; FN = 0
            acc,cost,y1,p1= sess.run([accuracy,loss,y_actual,y_pred], feed_dict={input_layer: dx, y: dy, freqFeat: dx1})     
            
            a,b,c,d=compute_pos_neg(y1,p1)
            TP+=a; FP+=b;TN+=c; FN+=d
            prec=metrics(TP,FP,TN,FN)
            
            print("Iter " + str(step*batch_size) + ", Loss= " + str(cost) +  \
              ", epoch= " + str(epoch_iter)+ \
              ", batch= "+ str(iter_tamp) +  ", acc= "+ str(acc)+ ", precision= "+str(prec))
              

        if step % 100== 0:
            TP = 0; FP = 0;TN = 0; FN = 0 
            num_images=batch_size
            n_chunks=(float)(np.shape(Img)[0]/batch_size)
            tAcc=np.zeros(n_chunks)

            for chunk in range(0,n_chunks):               
                tx_batch=Img[((chunk)*num_images):((chunk+1)*num_images),...]
                ty_batch=Lab[((chunk)*num_images):((chunk+1)*num_images),...]
                tx1_batch=freq1[((chunk)*num_images):((chunk+1)*num_images),...]
                ty_batch=conv_mask_gt(ty_batch)
                tAcc[chunk],y2,p2=sess.run([accuracy,y_actual,y_pred], feed_dict={input_layer: tx_batch, y:ty_batch, freqFeat: tx1_batch})
            a,b,c,d=compute_pos_neg(y2,p2)

            TP+=a; FP+=b;TN+=c; FN+=d
            
            prec=metrics(TP,FP,TN,FN)            
            test_accuracy=np.mean(tAcc)
           
            if prec > best_acc :
                best_prec = prec
                save_path=saver.save(sess,'model/final_model_nist.ckpt')
                #print("Best Model Found on NC16...")
            
            #print("prec = "+str(prec)+"("+str(best_prec)+")" + ", acc = "+ str(test_accuracy))
            
        step += 1

        if step % 500 ==0: 
            save_path=saver.save(sess,'model/final_model_nist.ckpt')
            print('model saved ..........#epoch->'+str(epoch_iter))
    #print("Optimization Finished!")
