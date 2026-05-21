import torch
import random
from torch.utils.data import Dataset

class EegDataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''

    def __init__(self, feature, label, subject_id=None):
        super(EegDataset, self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def get_num_class(self, num_class=[1, 1, 1, 1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class) > 0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label] > 0:
                num_class[label] -= 1
                res[label].append((self.x[i], self.y[i]))

        re2 = []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)

        y = torch.stack([x[1] for x in re2], dim=0)

        return x, y

    def get_num_subject(self, num_class=[1, 1, 1, 1, 1, 1, 1, 1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class) > 0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s] > 0:
                num_class[s] -= 1
                res[s].append((self.x[i], self.y[i]))

        re2 = []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)

        return x, y