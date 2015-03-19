import theano
import theano.tensor as T
import numpy as np
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import nose
import DeepLearningTB.DBN as db


class EmissionModel:

    #emission type: 0 - Matrix, 1 - MLP
    def __init__(self, emission, sequence, emission_type=0, phoneme_priors=None):
        
        self.seq = sequence
        self.emission_type = emission_type
        if (emission_type != 0):
            self.emission_probs = (emission.logLayer.p_y_given_x/ phoneme_priors)
            self.dbn = emission
        else: 
            self.emission_probs = emission
        
       
    
    # obs is the feature idx
    def emission_prob(self, t_idx, state=None):
        idx = t_idx if self.emission_type == 1 else self.seq[t_idx,0]
        if (state is not None):
            prob = self.emission_probs[idx,state]
        else:
            prob = self.emission_probs[idx,:]
        
        return prob
class HMM:
   '''
   Implementation of Hidden Markov Model using theano library.
   This model works in the log-likelihood domain to avoid risks with underflow 
   in long sequences.
   Reference paper : A tutorial on Hidden Markov Models and Selected 
   Applications in Speech Recognition, L.R. Rabiner  (1989) 
   Implementated by : Nizar Gandy Assaf Layouss. '''



   def __init__(self, n_states=None,n_observation=None,
                emission=None, initial=None, transition=None, phoneme_priors=None):
       #Matrix of stacked observations (for testing integer temp)
       self.x  = T.fmatrix('One Observation Sequence')
       

       if (n_states is not None and n_observation is not None):
                        
           initial = np.ones(n_states,dtype=theano.config.floatX)
           initial = initial/n_states
           
            #final = np.ones(n_states,dtype=theano.config.floatX)
            #final = final/n_states
           
           transition = np.ones([n_states + n_silent_states, n_states + n_silent_states], dtype=theano.config.floatX)
           transition = transition/np.expand_dims(np.sum(transition,axis=1),1)
           
           
           if (emission == None):
               emission = np.ones([n_states,n_observation], dtype=theano.config.floatX)
               emission = emission/np.expand_dims(np.sum(emission, axis=1),1)                 
               
            
                
                
    
       self.pi = theano.shared(initial, 'Initial transitions')
       #self.phi = theano.shared(final  , 'Final transitions')
       self.A  = theano.shared(transition, 'Transitions')

       if(not isinstance(emission, db.DBN)):
           self.B = EmissionModel(theano.shared(emission, 'Emission'),sequence=self.x)
       else:
           self.B = EmissionModel(emission,sequence=self.x ,emission_type=1,
                                   phoneme_priors=theano.shared(phoneme_priors,'Phoneme priors'))

       self.n_states        = self.A.eval().shape[0]
       #self.n_silent_states = n_silent_states
       self.epsilon_hist = theano.shared(np.zeros((self.n_states,self.n_states),dtype=theano.config.floatX))
       self.gamma_hist   = theano.shared(np.zeros((self.n_states,1),dtype=theano.config.floatX))
       self.last_A = None
   
   def save_last_A(self):
       self.last_A = theano.shared(np.array(self.A.get_value(borrow=False),dtype=theano.config.floatX))
   
   def load_last_A(self):
       if self.last_A is not None:
           self.A.set_value(self.last_A.get_value())
   
   def set_emission_fn(self, emission):
       self.emission = emission
    
   def forward_trellis(self):
       #Calculates forward for one step time t
       def scan_step(t,last_alpha,last_coeff):
           alpha_t = T.dot(last_alpha, self.A)* self.B.emission_prob(t_idx=t)     #self.B[:,obs_seq[t,0]]
           coeff   = 1. /T.sum(alpha_t)
           return [alpha_t*coeff, coeff]
            
       initial = self.pi*self.B.emission_prob(t_idx=0)       #self.B[:,self.x[0,0]]
       coef    = T.cast(1. /T.sum(initial), dtype=theano.config.floatX)
       initial_step = [initial*coef,coef]
       #print initial_step.type
       components, updates = theano.scan(fn=scan_step,
                                         outputs_info=initial_step,
                                         sequences=[T.arange(self.x.shape[0]-1)+1])

       
       return [T.concatenate([[initial_step[0]],components[0]]),T.concatenate([[initial_step[1]],components[1]])]

   def backward_trellis(self, coeff):
        
       def scan_step(t, previous_beta, coeff):
           return  coeff[t]*T.dot(self.A, self.B.emission_prob(t_idx=t+1)*previous_beta)
       
       initial_step = coeff[0]*T.as_tensor_variable(np.ones(self.n_states,dtype=theano.config.floatX))
       
       components, updates = theano.scan(fn=scan_step,
                                         outputs_info=initial_step,
                                         sequences=[T.arange(self.x.shape[0]-1,0,-1)-1],
                                         non_sequences=[coeff])
       
       return T.concatenate([components[::-1],[initial_step]])


   def build_forward_fn(self, sequences, idces):
       
       def scan_step(t,last_alpha):
           alpha_t = T.dot(last_alpha ,self.A)* self.B.emission_prob(t_idx=t)      #self.B[:,obs_seq[t,0]]
           return alpha_t
       
       initial = self.pi*self.B.emission_prob(t_idx=0)       #self.B[:,self.x[0,0]]
       components, updates = theano.scan(fn=scan_step,
                                         outputs_info=initial,
                                         sequences=[T.arange(self.x.shape[0]-1)+1])
        
       p = T.sum(components[-1])
       index = T.lscalar("Utterance Index")
       if (self.B.emission_type):
           #case DNN
           return theano.function(inputs=[index], 
                                  outputs=p, 
                                  givens={self.x:sequences[idces[index,0]:idces[index,1]+1],
                                          self.B.dbn.x:sequences[idces[index,0]:idces[index,1]+1]})
       
       else:
           return theano.function(inputs=[index], 
                                  outputs=p, 
                                  givens={self.x:sequences[idces[index,0]:idces[index,1]+1]})
       
       
   def build_viterbi_fn(self, sequences, idces):

       

       #Calculates viterbi for one step time t
       def scan_step(t,last_trellis):
           return T.max(last_trellis.reshape((self.n_states,1))+T.log(self.A),axis=0)+T.log(self.B.emission_prob(t_idx=t))
       
       
       #print self.B.emission_prob(obs_idx=0).eval()
       initial_step = T.log(self.pi)+T.log(self.B.emission_prob(t_idx=0))
       
       
       components, updates = theano.scan(fn=scan_step,
                                         outputs_info=initial_step,
                                         sequences=[T.arange(self.x.shape[0]-1)+1])
                                         
       
       trellis = T.concatenate([[initial_step],components])
       
       #Final transition
       logp = T.max(trellis[-1,:])
       path = T.argmax(trellis,axis=1)
       
       index = T.lscalar("Utterance Index")
       
       if (self.B.emission_type):
           #DNN case
           return theano.function(inputs=[index],outputs=[path,logp],updates=updates,
                                  givens={self.x:sequences[idces[index,0]:idces[index,1]+1],
                                          self.B.dbn.x:sequences[idces[index,0]:idces[index,1]+1]})
       else:
           return theano.function(inputs=[index],outputs=[path],updates=updates,
                                  givens={self.x:sequences[idces[index,0]:idces[index,1]+1]})
       
   def reset_historial(self):
       
       self.epsilon_hist.set_value(np.zeros((self.n_states,self.n_states),dtype=theano.config.floatX))
       self.gamma_hist.set_value(np.zeros((self.n_states,1),dtype=theano.config.floatX))
       




   def build_train_fn(self, sequences, idces):
              
       # foward pass
       alpha,coeff = self.forward_trellis()
       
       # backward pass 
       beta = self.backward_trellis(coeff)


       #Computing epsilon
       def epsilon_step(t, alpha, beta):
           epsilon_t = self.B.emission_prob(t_idx=t+1)*beta[t+1,:]*(alpha[t,:].reshape((self.n_states,1))*self.A)
           #epsilon_t = epsilon_t / T.sum(epsilon_t)
           return epsilon_t
        

       components_epsilon, updates_epsilon = theano.scan(fn=epsilon_step,
                                                         outputs_info=None,
                                                         sequences=[T.arange(self.x.shape[0]-1)],
                                                         non_sequences=[alpha, beta])
       
       epsilon = components_epsilon
       epsilon_acc = T.sum(epsilon ,axis=0)
       
       gamma = (alpha*beta)/coeff.reshape((self.x.shape[0],1))
       #gamma = gamma / T.sum(gamma,axis=1).reshape((self.x.shape[0],1))
       
       index = T.lscalar("Utterance Index")
       
       

       logp = -T.sum(T.log(coeff))
       #re-estimate new paramters
       gamma_sum = T.sum(gamma[0:-1,:], axis=0).reshape((self.n_states,1))
       
       #new_A = (self.epsilon_hist + epsilon_acc)/(self.gamma_hist + gamma_sum).reshape((self.n_states,1))
       
   
       new_A = (self.epsilon_hist)/(self.gamma_hist).reshape((self.n_states,1))

       update_fn = theano.function(inputs=[],
                                   outputs=[],
                                   updates=[(self.A ,new_A)])

       if (not self.B.emission_type):
           #Add reestimation of B matrix TODO !!
           updates = [(self.pi, gamma[0,:]), 
                      #(self.A, new_A ),
                      (self.epsilon_hist,self.epsilon_hist + epsilon_acc),
                      (self.gamma_hist,self.gamma_hist + gamma_sum)]
           
           return theano.function(inputs=[index],
                                  outputs=[logp],
                                  updates=updates,
                                  givens={self.x:sequences[idces[index,0]:idces[index,1]+1]})
           
       else:
           #case DNN
           updates = [(self.epsilon_hist,self.epsilon_hist + epsilon_acc),
                      (self.gamma_hist,self.gamma_hist + gamma_sum)]
           
           return theano.function(inputs=[index],
                                  outputs=[logp],
                                  updates=updates,
                                  givens={self.x:sequences[idces[index,0]:idces[index,1]+1],
                                          self.B.dbn.x:sequences[idces[index,0]:idces[index,1]+1]}),update_fn     
           
                                      

def test_0():
    Pi = np.asarray([0.4, 0.3, 0.3], dtype=theano.config.floatX)
    Trans = np.asarray([[0.4, 0.4, 0.2], 
                        [0.3, 0.6, 0.1],
                        [0.3, 0.2, 0.5]], dtype=theano.config.floatX)
    Emission = np.asarray([[0.3, 0.2, 0.1, 0.4],    
                           [0.1, 0.3, 0.4, 0.2],    
                           [0.4, 0.1, 0.3, 0.2]], dtype=theano.config.floatX)

    #Final = np.array([1/3., 1/3., 1/3.], dtype=theano.config.floatX) 
    obs = theano.shared(np.asmatrix([[1],[0], [2],[2],[3],[1],[0], [2],[2],[3]], dtype='int32'))
    idces = theano.shared(np.asarray([[0,4],[5,9]], dtype='int32'))
    hmm = HMM(initial=Pi, transition=Trans, emission=Emission)
    
    v_fn       = hmm.build_viterbi_fn(obs,idces)
    forward_fn = hmm.build_forward_fn(obs,idces)
    resv=v_fn(0)
    p_forward  = forward_fn(0) 
  
    

    assert (resv[0] + 10.0259532928) < 1e-06
    
    assert (p_forward - 9.0421e-04) < 1e-03
    assert tuple(resv[1]) == (1, 0, 1, 1, 0)
    
   
  
def test_1():
    Pi = np.asarray([0.4, 0.3, 0.3], dtype=theano.config.floatX)
    Trans = np.asarray([[0.4, 0.4, 0.2], 
                        [0.3, 0.6, 0.1],
                        [0.3, 0.2, 0.5]], dtype=theano.config.floatX)
    Emission = np.asarray([[0.3, 0.2, 0.1, 0.4],    
                           [0.1, 0.3, 0.4, 0.2],    
                           [0.4, 0.1, 0.3, 0.2]], dtype=theano.config.floatX)
    
    #Final = np.array([1/3., 1/3., 1/3.], dtype=theano.config.floatX) 
    obs = theano.shared(np.asmatrix([[1],[0], [2],[2],[3],[1],[0], [2],[2],[3]], dtype='int32'))
    idces = theano.shared(np.asarray([[0,4],[5,9]], dtype='int32'))
    hmm = HMM(initial=Pi, transition=Trans, emission=Emission)
    
    train_fn   = hmm.build_train_fn(obs,idces)
    train_fn(0)
    
    mpi = np.asarray(hmm.pi.eval())
    assert ((mpi - np.asarray([ 0.41871244,  0.38488674,  0.19640084]
                              ,dtype = theano.config.floatX)) < 1e-06*np.ones(mpi.shape,dtype = theano.config.floatX)).all()
    mA = np.asarray(hmm.A.eval())
    assert ((mA -  np.asarray([[ 0.32444078, 0.4307338,   0.2448253 ],
                               [ 0.29849356, 0.59171104,  0.10979547],
                               [ 0.23073582, 0.23792365,  0.53134054]], 
                              dtype = theano.config.floatX))  <  1e-06*np.ones(mA.shape,dtype = theano.config.floatX)).all()
    
    #print hmm.pi.eval()
    #print hmm.A.eval()
