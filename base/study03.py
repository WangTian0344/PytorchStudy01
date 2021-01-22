import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor)

print("tensor:{},variable:{}".format(tensor,variable))
print(variable.data)