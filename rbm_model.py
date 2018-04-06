import tensorflow as tf
import model_helper

class RBM:
    ''' Implementation of the Restricted Boltzmann Machine for collaborative filtering. The model is based on the paper of 
        Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton: https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
    '''
    def __init__(self, FLAGS):
        '''Initialization of the model  '''
        self.FLAGS=FLAGS
        self.weight_initializer=model_helper._get_weight_init()
        self.bias_initializer=model_helper._get_bias_init()
        self.init_parameter()
        

    def init_parameter(self):
        ''' Initializes the weights and the bias parameters of the neural network.'''

        with tf.variable_scope('Network_parameter'):
            self.W=tf.get_variable('Weights', shape=(self.FLAGS.num_v, self.FLAGS.num_h),initializer=self.weight_initializer)
            self.bh=tf.get_variable('hidden_bias', shape=(self.FLAGS.num_h), initializer=self.bias_initializer)
            self.bv=tf.get_variable('visible_bias', shape=(self.FLAGS.num_v), initializer=self.bias_initializer)
            
    
    def _sample_h(self, v):
        ''' Uses the visible nodes for calculation of  the probabilities that a hidden neuron is activated. 
        After that Bernouille distribution is used to sample the hidden nodes.
        
        @param v: visible nodes
        @return probability that a hidden neuron is activated
        @return sampled hidden neurons (value 1 or 0 accroding to Bernouille distribution)
        '''
        
        with tf.name_scope('sampling_hidden_units'):

            a=tf.nn.bias_add(tf.matmul(v,self.W), self.bh)
            p_h_v=tf.nn.sigmoid(a)
    
            bernouille_dis=tf.where(tf.less(p_h_v, tf.random_uniform(shape=[self.FLAGS.batch_size,self.FLAGS.num_h],minval=0.0,maxval=1.0)),
                                    x=tf.zeros_like(p_h_v),
                                    y=tf.ones_like(p_h_v))
            return p_h_v, bernouille_dis
        
    
    def _sample_v(self, h):
        ''' Uses the hidden nodes for calculation of  the probabilities that a visible neuron is activated. 
        After that Bernouille distribution is used to sample the visible nodes.
        
        @param h: hidden nodes
        @return probability that a visible neuron is activated
        @return sampled visible neurons (value 1 or 0 accroding to Bernouille distribution)
        '''
        
        with tf.name_scope('sampling_visible_units'):
            a=tf.nn.bias_add(tf.matmul(h,tf.transpose(self.W, [1,0])), self.bv)
            p_v_h=tf.nn.sigmoid(a)
    
            bernouille_dis=tf.where(tf.less(p_v_h, tf.random_uniform(shape=[self.FLAGS.batch_size,self.FLAGS.num_v],minval=0.0,maxval=1.0)),
                                    x=tf.zeros_like(p_v_h),
                                    y=tf.ones_like(p_v_h))
            return p_v_h, bernouille_dis
    
    
    def _optimize(self, v):
        ''' Optimization step. Does Gibbs sampling, calculates gradiends and produces update operations.
         Also calculates accuracy the training data.
        
        @param v: visible nodes
        @return update operation
        @return accuracy
        '''

        with tf.name_scope('optimization'):
            v0, vk,ph0, phk, _=self._gibbs_sampling(v)
            dW,db_h,db_v=self._compute_gradients(v0, vk, ph0, phk)
            update_op =self._update_parameter(dW,db_h,db_v)
        
        with tf.name_scope('accuracy'):
            mask=tf.where(tf.less(v0,0.0),x=tf.zeros_like(v0),y=tf.ones_like(v0))
            bool_mask=tf.cast(tf.where(tf.less(v0,0.0),x=tf.zeros_like(v0),y=tf.ones_like(v0)), dtype=tf.bool)
            acc=tf.where(bool_mask,x=tf.abs(tf.subtract(v0,vk)),y=tf.zeros_like(v0))
            n_values=tf.reduce_sum(mask)
            acc=tf.subtract(1.0,tf.div(tf.reduce_sum(acc), n_values))
            
        return update_op, acc
    
    def _inference(self, v):
        '''Inference step. Visible training nodes are given to activate the hidden neurons. Hidden neurons are then used 
        to activate visible neurons.
        
        @param v: visible nodes
        @return sampled visible neurons (value 1 or 0 accroding to Bernouille distribution)
        '''
        
        p_h_v=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(v,self.W), self.bh))
        bernouille_dis=tf.where(tf.less(p_h_v, tf.random_uniform(shape=[1,self.FLAGS.num_h],minval=0.0,maxval=1.0)),
                                x=tf.zeros_like(p_h_v),
                                y=tf.ones_like(p_h_v))
        
       
        p_v_h=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(bernouille_dis,tf.transpose(self.W, [1,0])), self.bv))
    
        bernouille_dis=tf.where(tf.less(p_v_h, tf.random_uniform(shape=[1,self.FLAGS.num_v],minval=0.0,maxval=1.0)),
                                x=tf.zeros_like(p_v_h),
                                y=tf.ones_like(p_v_h))
        return bernouille_dis
        
    
    def _update_parameter(self,dW,db_h,db_v):
        ''' Creating TF assign operations. Updated weight and bias values are replacing old parameter values.
        
        @return assign operations
        '''
        
        alpha=self.FLAGS.learning_rate
        
        update_op=[tf.assign(self.W, alpha*tf.add(self.W,dW)),
                   tf.assign(self.bh, alpha*tf.add(self.bh,db_h)),
                   tf.assign(self.bv, alpha*tf.add(self.bv,db_v))]

        return update_op
    
    
    def _compute_gradients(self,v0, vk, ph0, phk):
        ''' Computing the gradients of the weights and bias terms with Contrastive Divergence.
        
        @param v0: visible neurons before gibbs sampling
        @param vk: visible neurons before gibbs sampling
        @param ph0: probability that hidden neurons are activated before gibbs sampling.
        @param phk: probability that hidden neurons are activated after gibbs sampling.
        
        @return gradients of the network parameters
        
        '''

        def condition(i, v0, vk, ph0, phk, dW,db_h,db_v):
            r=tf.less(i,k)
            return r[0]
        
        def body(i, v0, vk, ph0, phk, dW,dbh,dbv):
            
            v0_=v0[i]
            ph0_=ph0[i]
            
            vk_=vk[i]
            phk_=phk[i]       
            
            ph0_=tf.reshape(ph0_, [1,self.FLAGS.num_h])
            v0_=tf.reshape(v0_, [self.FLAGS.num_v,1])
            
            phk_=tf.reshape(phk_, [1,self.FLAGS.num_h])
            vk_=tf.reshape(vk_, [self.FLAGS.num_v,1])
            
            dw_=tf.subtract(tf.multiply(ph0_, v0_),tf.multiply(phk_, vk_))
            dbh_=tf.subtract(ph0_,phk_)
            dbv_=tf.subtract(v0_,vk_)
            
            dbh_=tf.reshape(dbh_,[self.FLAGS.num_h])
            dbv_=tf.reshape(dbv_,[self.FLAGS.num_v])
            
            return [i+1, v0, vk, ph0, phk,tf.add(dW,dw_),tf.add(dbh,dbh_),tf.add(dbv,dbv_)]
        
        i = 0
        
        k=tf.constant([self.FLAGS.batch_size])
        dW=tf.zeros((self.FLAGS.num_v, self.FLAGS.num_h))
        dbh=tf.zeros((self.FLAGS.num_h))
        dbv=tf.zeros((self.FLAGS.num_v))
        
        [i, v0, vk, ph0, phk, dW,db_h,db_v]=tf.while_loop(condition, body,[i, v0, vk, ph0, phk, dW,dbh,dbv])
          
        dW=tf.div(dW, self.FLAGS.batch_size)
        dbh=tf.div(dbh, self.FLAGS.batch_size)
        dbv=tf.div(dbv, self.FLAGS.batch_size)
        
        return dW,dbh,dbv
        
    def _gibbs_sampling(self, v):
        ''' Perfroming the gibbs sampling.
        
        @param v: visible neurons
        @return visible neurons before gibbs sampling
        @return visible neurons before gibbs sampling
        @return probability that hidden neurons are activated before gibbs sampling.
        @return probability that hidden neurons are activated after gibbs sampling.
        '''
 
        def condition(i, vk, hk,v):
            r= tf.less(i,k)
            return r[0]
        
        def body(i, vk, hk,v):
            
            _,hk=self._sample_h(vk)
            _,vk=self._sample_v(hk)

            vk=tf.where(tf.less(v,0),v,vk)
            
            return [i+1, vk, hk,v]
            
        ph0,_=self._sample_h(v)
        
        vk=v
        hk=tf.zeros_like(ph0)
            
        i = 0
        k=tf.constant([self.FLAGS.k])
        
        [i, vk,hk,v]=tf.while_loop(condition, body,[i, vk,hk,v])
        
        phk,_=self._sample_h(vk)
        
        return v, vk,ph0, phk, i
        
        
        
        
        
        
        
        
        
        
        
        
        
    