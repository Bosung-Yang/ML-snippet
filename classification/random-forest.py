from sklearn.ensemble import RandomForestClassifier

def init_model():
  clf = RandomForestClassifier(random_state = 42)
  return clf

def train(model, x, y):
  clf.fit(x,y)
  return clf

from sklearn.metrics import *

def evaluate(clf, x, y):
  predicted = clf.predict(x)
  print('[F1 score] : ',f1_score(y, pred, average='macro'))
  print('[Acc] : ', accuracy_score(y, pred))
  print('[Precision] : ', precision_score(y, pred, average='macro'))
  print('[Recall] : ', recall_score(y,pred, average='macro'))

import pickle

def save_model(clf, fn): # clf : classification model, fn : file name
  try:
    with open(fn, 'wb') as f:
      pickle.dump(clf, f)
    print('[+] model saved {}'.format(f))
    return True
  except:
    print('[!] error at save_model')
    return False

def load_model(fn):
  try:
    with open(fn, 'rb') as f:
      clf = pickle.load(f)
      print('[+] {} is loaded'.format(f))
      return clf
  except:
    print('[!] error at load_model')
    return False
    
