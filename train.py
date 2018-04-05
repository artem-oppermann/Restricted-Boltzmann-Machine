import tensorflow as tf
from data.dataset import _get_training_data, _get_test_data
from rbm_model import RBM
import numpy as np



tf.app.flags.DEFINE_string('tf_records_train_path', 'C:/Users/Admin/Desktop/deep_learning _local_datasets/colaborative filtering/tf_records_100k/train/',
                           'Path of the training data.'
                           )

tf.app.flags.DEFINE_string('tf_records_test_path', 'C:/Users/Admin/Desktop/deep_learning _local_datasets/colaborative filtering/tf_records_100k/test/',
                           'Path of the test data.'
                           )

tf.app.flags.DEFINE_integer('num_epoch', 1000,
                            'Number of training epochs.'
                           )
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'Size of the training batch.'
                           )

tf.app.flags.DEFINE_float('learning_rate',1.0,
                          'Learning_Rate')


tf.app.flags.DEFINE_integer('num_v', 1682,
                            'Number of visible neurons.'
                            )

tf.app.flags.DEFINE_integer('num_h', 100,
                            'Number of hidden neurons.'
                            )


tf.app.flags.DEFINE_integer('num_samples', 943,
                            'Number of training samples.'
                            )


tf.app.flags.DEFINE_integer('k', 10,
                           'Number of iterations during gibbs samling.'
                           )

tf.app.flags.DEFINE_integer('eval_after',25,
                            'Evaluate model after number of iterations.'
                           )

FLAGS = tf.app.flags.FLAGS



def main(_):
    
    num_batches=int(FLAGS.num_samples/FLAGS.batch_size)

    with tf.Graph().as_default():

        train_data, train_data_infer=_get_training_data(FLAGS)
        test_data=_get_test_data(FLAGS)
        
        iter_train = train_data.make_initializable_iterator()
        iter_train_infer = train_data_infer.make_initializable_iterator()
        iter_test = test_data.make_initializable_iterator()
        
        x_train= iter_train.get_next()
        x_train_infer=iter_train_infer.get_next()
        x_test=iter_test.get_next()
        
        model=RBM(FLAGS)

        update_op, accuracy=model._optimize(x_train)
        v_infer=model._inference(x_train_infer)
 
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.num_epoch):
                
                acc_train=0
                acc_infer=0

                sess.run(iter_train.initializer)
                
                for batch_nr in range(num_batches):
                    _, acc=sess.run((update_op, accuracy))
                    acc_train+=acc
                
                    if batch_nr>0 and batch_nr%FLAGS.eval_after==0:
                        
                        sess.run(iter_train_infer.initializer)
                        sess.run(iter_test.initializer)

                        num_valid_batches=0
                    
                        for i in range(FLAGS.num_samples):
    
                            v_target=sess.run(x_test)[0]
                    
                            if len(v_target[v_target>=0])>0:
                                    
                                v_=sess.run(v_infer)[0]
                                acc=1.0-np.mean(np.abs(v_[v_target>=0]-v_target[v_target>=0]))
                                acc_infer+=acc
                                num_valid_batches+=1
        
                        print('epoch_nr: %i, batch: %i/%i, acc_train: %.3f, acc_test: %.3f'%
                              (epoch, batch_nr, num_batches, (acc_train/FLAGS.eval_after), (acc_infer/num_valid_batches)))
                    
                        acc_train=0
                        acc_infer=0

    
        
if __name__ == "__main__":
    
    tf.app.run()

    
  



