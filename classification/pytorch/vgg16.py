import torch 
import torchvision
import numpy as np 
from torchvision import transforms, datasets
import torch.nn.functional as F

def dataset_transforms():
  image_transforms = {
    'train': transforms.Compose([
            transforms.CenterCrop((128,128)),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
  }
  return image_transforms

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_custom_dataloader():
  image_transforms = dataset_transforms()

  data_path = ''
  train_folder = datapath + '/train/'
  test_folder = datapath + '/test/'
  train_images = datasets.ImageFolder(data_path+train_folder,image_transforms['train'])
  train_loader = torch.utils.data.DataLoader(train_images, batch_size = 64 ,num_workers=4,shuffle=True)
  test_images = datasets.ImageFolder(data_path+test_folder,image_transforms['test'])
  test_loader = torch.utils.data.DataLoader(test_images, batch_size = 64 ,num_workers=4,shuffle=True) 
  return train_loader, test_loader

def get_model():
  return torchvision.models.vgg16_bn(pretrained=True)

def get_objective(fn = 'ce'):
  if fn == 'ce': return torch.nn.CrossEntropyLoss()


def train(model,train_loader, eval_loader, epoch):
  objective = get_objective()
  lr = 0.0001
  torch.optim.Adam(params=model.parameters(), lr=lr)
  verbose = True
  for e in range(epoch):
    for _, data in enumerate(train_loader):
      model.train()
      optimizer.zero_grad()
      img, label = data
      img = img.cuda()
      label = label.cuda()
      predicted = model(img)
      loss = objective(label, predicted)
      loss.backward()
      optimizer.step()
    
    if verbose:
      for _, data in enumerate(eval_loader):
          img, label = data
          img = img.to(device)
          label = label.to(device)
          predicted = model(img)
          _, p_label = predicted.max(1)
          total += label.size(0)
          answer += (p_label == label).sum().float()
      print('Accuracy : {:.4f}'.format(answer/total))
      
def main(args = None):
  set_seed(42)
  train_loader, test_loader = get_custom_dataloader()
  model = get_model()
  train(model, train_loader, test_loader, 100)
  
          

  
