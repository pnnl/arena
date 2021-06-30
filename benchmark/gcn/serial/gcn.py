import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')


# customized kernel preparing for offloading
def customMM(x, weight):
#  out = [[0 for _ in weight[0]] for _ in x]
#  print("out: ", len(out), len(out[0]))
#  for i in range(len(x)):
#    for j in range(len(weight[0])):
#      for k in range(len(x[0])):
#        out[i][j] += x[i][k] * weight[k][j]
  A = np.array(x)
  B = np.array(weight)
  out = A.dot(B)
  return out

def customAdd(x, bias):
  for i in range(len(x)):
    for j in range(len(x[0])):
      x[i][j] += bias[j]
  return x
      
def customRelu(x):
  for i in range(len(x)):
    for j in range(len(x[0])):
      if x[i][j] < 0:
        x[i][j] = 0
  return x
 
def customBuildEdge(edge, m, n):
  out = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(len(edge[0])):
    out[edge[0][i]][edge[1][i]] = 1
  return out


def customEqual(a, b):
  if len(a) != len(b):
    return False
  if len(a[0]) != len(b[0]):
    return False
  for i in range(len(a)):
    for j in range(len(a[0])):
      if abs(a[i][j] - b[i][j]) > 0.001:
        print(i, j, a[i][j], b[i][j])
        return False
  return True

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16, normalize=False)
        self.conv2 = GCNConv(16, dataset.num_classes, normalize=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        print("[example] feature ", len(x), " x ", len(x[0]), ": ")
        print(x)
        print("\n------------------------------ init finished -----------------------------------\n") 

        x = self.conv1(x, edge_index)
        print("[example] conv1 result ", len(x), " x ", len(x[0]), ": ")
        print(x)
        print("\n------------------------------ conv1 finished ----------------------------------\n") 

        x = F.relu(x)
        print("[example] relu: ")
        print(x)
        print("\n------------------------------- relu finished ----------------------------------\n") 

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        print("[example] out (", len(x), "x", len(x[0]), "):")
        print(x)
        print("\n------------------------------ conv2 finished ---------------------------------\n")

        # essential for training
        # return F.log_softmax(x, dim=1)
        return x

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--train', action='store_true', default=False,
                    help='train network')
args = parser.parse_args()
PATH = "./weights.pt"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = Net().to(device)
data = dataset[0].to(device)

if args.train:
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

  model.train()
  for epoch in range(200):
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()

  torch.save(model.state_dict(), PATH)
  # print(model.state_dict())
  print("[example] train GCN and store weights into ", PATH)

else:
  print("\n============================ custom offload =================================\n") 
  model.load_state_dict(torch.load(PATH))
  print("[custom] load weights: ")
  print( model.state_dict())
  weight1 = model.state_dict()["conv1.weight"]
  bias1 = model.state_dict()["conv1.bias"]
  weight2 = model.state_dict()["conv2.weight"]
  bias2 = model.state_dict()["conv2.bias"]
  print("[custom] weight1 ", len(model.state_dict()["conv1.weight"].numpy()), " x ", len(model.state_dict()["conv1.weight"][0].numpy()), ": ")
  print(weight1)

  # Custom computation for offload and verification.
  edgeMatrix = customBuildEdge(data.edge_index, len(data.x), len(data.x))
  print("\n------------------------ weight loading finished ----------------------------\n")
  print("[custom] conv1 mm: ")
  mm = customMM(data.x, weight1)
  print("..see mm: ")
  print(mm)
  mm = customMM(edgeMatrix, mm)
  mm = customAdd(mm, bias1)
  mm = customRelu(mm);
  print("..final: ")
  print(mm)
  print("\n------------------------- conv1 & relu finished ------------------------------\n")
  mm = customMM(edgeMatrix, data.x)
  mm = customMM(mm, weight1)
  mm = customAdd(mm, bias1)
  mm = customRelu(mm);
  print("[custom] alternative conv1 mm: ")
  print(mm)
  print("\n------------------- alternative conv1 & relu finished ------------------------\n")

  mm = customMM(edgeMatrix, mm)
  mm = customMM(mm, weight2)
  mm = customAdd(mm, bias2)
  print("[custom] conv2 mm: ")
  print(mm)
  print("\n----------------------------- conv2 finished ----------------------------------\n")

  print("\n============================== GCN geometric ==================================\n")
  model.eval()
  result = model(data)
  _, pred = result.max(dim=1)
  if customEqual(mm, result):
    print("[offload] success!")
    print(result)
  else:
    print("[offload] fail!")
  correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
  acc = correct / data.test_mask.sum().item()

  print('[example] test GCN and accuracy is: {:.4f}'.format(acc))
