import pandas as pd
import numpy as np
import gc

ROOT_DIR='C:/Users/Admin/Desktop/deep_learning _local_datasets/colaborative filtering'


def convert(data, num_users, num_movies):
    
    new_data=[]
    
    for id_user in range(1, num_users+1):
        print(id_user)
        id_movie=data[:,1][data[:,0]==id_user]
        id_rating=data[:,2][data[:,0]==id_user]
        ratings=np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie-1]=id_rating
        new_data.append(ratings)
        
        del id_movie
        del id_rating
        del ratings
        
    return new_data

def get_dataset_1M():
    
    gc.enable()
    
    training_set=pd.read_csv(ROOT_DIR+'/ml-10M/r1.train', sep='::', header=None, engine='python', encoding='latin-1')
    training_set=np.array(training_set, dtype=np.uint32)
    
    test_set=pd.read_csv(ROOT_DIR+'/ml-10M/r1.test', sep='::', header=None, engine='python', encoding='latin-1')
    test_set=np.array(test_set, dtype=np.uint32)
      
    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))
    
    print(num_users)
    print(num_movies)
    
    training_set=convert(training_set,num_users, num_movies)
    print('hier')
    test_set=convert(test_set,num_users, num_movies)
    
    return training_set, test_set
    


def get_dataset_100k():
    
    training_set=pd.read_csv(ROOT_DIR+'/ml-100k/ua.base', delimiter='\t')
    training_set=np.array(training_set, dtype=np.uint32)
    
    test_set=pd.read_csv(ROOT_DIR+'/ml-100k/ua.test', delimiter='\t')
    test_set=np.array(test_set, dtype=np.uint32)
      
    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))
    
    print(num_users)
    print(num_movies)
    
    training_set=convert(training_set,num_users, num_movies)
    print('hier')
    test_set=convert(test_set,num_users, num_movies)
    
    return training_set, test_set



def _get_dataset(kind):

    if kind=='1M':
        training_set, test_set=get_dataset_1M()
    elif kind=='100k':
        training_set, test_set=get_dataset_100k()
    
    return training_set, test_set


training_set, test_set=_get_dataset('1M')


