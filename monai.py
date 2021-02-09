import os
import argparse
import copy
import pandas as pd
import pickle
import time
from PIL import Image
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from utils import progress_bar
from scipy.ndimage import rotate


###IGNITE Configs
import logging
import sys
from ignite.metrics import Accuracy


### MONAI Configs
import monai
from monai.engines import SupervisedTrainer, SupervisedEvaluator, Trainer
from monai.config import print_config
from monai.transforms import Compose, AddChannel, ScaleIntensity, ToTensor, Transpose, RandRotate, RandFlip, RandZoom, AsDiscreted, ToTensord
from monai.networks.nets import densenet121
from monai.metrics import compute_roc_auc
from monai.handlers import CheckpointSaver, MetricLogger, LrScheduleHandler, StatsHandler, TensorBoardImageHandler, TensorBoardStatsHandler, ValidationHandler
from monai.inferers import SimpleInferer, SlidingWindowInferer

# early stopping with ignite
from monai.handlers import stopping_fn_from_metric
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.param_scheduler import LRScheduler


print_config()
parser = argparse.ArgumentParser(description='DMIST Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-m', type= str, default='resnet', help='Architecture')
parser.add_argument('--dataset', '-d', type= str, default='dmist2', help='Dataset to be trained on - dmist2, dmist3, dmist4, mgh, dmist')
parser.add_argument('--seed', '-s', type= int, default=0, help='seed')
parser.add_argument('--monai_trainer', action='store_true', help='whether to use monai trainers or not')
parser.add_argument('--percent_data', '-p', type= int, default=100, help='data percentage to be used, set 100 for full dataset training')
args = parser.parse_args()

### Set Seed globally
seed = args.seed
monai.utils.set_determinism(seed=seed, additional_settings=None)


replica_name = '_seed_'+str(seed)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
base_direc = os.getcwd()
results_folder = "/Results/LocalTraining/"+(args.dataset + '_' + args.model+'_MonaiTrainers_'+str(args.monai_trainer)+'_seed_'+str(args.seed)+'_percent_data_'+str(args.percent_data))+"/"
if not os.path.isdir(base_direc + results_folder):
	os.makedirs(base_direc + results_folder)

batch_size=32
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


class Dataset(data.Dataset):
	def __init__(self, list_IDs, transforms=None):
		self.list_IDs = list_IDs
		self.transforms = transforms

	def __getitem__(self, index):
		image_path = self.list_IDs[index]
		my_path=image_path.replace("/home/ken.chang@ccds.io/mnt","/data" )
		label = int(image_path[-5])-1
		return {"image": self.transforms(np.load(my_path).astype('float32') ),  "label": torch.from_numpy(np.array(label))} #LoadNumpy Not available in MONAI

	def __len__(self):
		return int(len(self.list_IDs))

#
os.chdir('/data/2015P002510/Sharut/Updated_Dataset/')
with open("Train_images_DMIST2.txt", "rb") as fp:
	Train_files1 = pickle.load(fp)
with open("Train_images_DMIST3.txt", "rb") as fp:
	Train_files2 = pickle.load(fp)
with open("Train_images_DMIST4.txt", "rb") as fp:
	Train_files3 = pickle.load(fp)
with open("Train_images_MGH.txt", "rb") as fp:
	Train_files4 = pickle.load(fp)

#
with open("Val_images_DMIST2.txt", "rb") as fp:
	Val_files1 = pickle.load(fp)
with open("Val_images_DMIST3.txt", "rb") as fp:
	Val_files2 = pickle.load(fp)
with open("Val_images_DMIST4.txt", "rb") as fp:
	Val_files3 = pickle.load(fp)
with open("Val_images_MGH.txt", "rb") as fp:
	Val_files4 = pickle.load(fp)


dataset_dict = {'dmist2':[Train_files1,Val_files1 ], 'dmist3':[Train_files2,Val_files2], 'dmist4':[Train_files3,Val_files3 ], 'mgh':[Train_files4,Val_files4], 'dmist':[Train_files1+Train_files2+Train_files3,Val_files1+Val_files2+Val_files3]}

Train_files , Val_files  = dataset_dict[args.dataset][0], dataset_dict[args.dataset][1]

num_size = (int)(args.percent_data * len(Train_files) / 100)
print("Original Length: ", len(Train_files), "Final Length: ", num_size)
Train_files = random.sample(Train_files, num_size)


num_size = (int)(args.percent_data * len(Val_files) / 100)
print("Original Length: ", len(Val_files), "Final length: ", num_size)
Val_files = random.sample(Val_files, num_size)


random.shuffle(Train_files)
random.shuffle(Val_files)


train_transforms = Compose([
    # monai.transforms.io.array.LoadNumpy(data_only=True, dtype= torch.float),
    # monai.transforms.LoadPNG(image_only=False),
    Transpose((2,0,1)),
    RandRotate(range_x=45, prob=1, keep_size=True, mode='nearest'), #2D Image here
    RandFlip(spatial_axis=0, prob=0.5),
    RandFlip(spatial_axis=1, prob=0.5),
    ToTensor()
])

val_transforms = Compose([
	Transpose((2,0,1)),
    ToTensor()
])


# LOADING DATASET 
training_set = Dataset(list_IDs=Train_files, transforms = train_transforms)
trainloader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

test_set = Dataset(list_IDs = Val_files, transforms = val_transforms)
testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)


# BUILDING MODEL & SENDING TO GPU

if args.model == 'resnet': #Not available in MONAI
	base_model = torchvision.models.resnet50(pretrained=True)
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Linear(num_ftrs, 4)
	print("Training on ResNet")

elif args.model == 'vgg19': #Not available in MONAI
	base_model = torchvision.models.vgg19(pretrained=True)
	num_ftrs = base_model.classifier[6].in_features
	base_model.classifier[6] = nn.Linear(num_ftrs, 4)
	print("Training on VGG-19")

elif args.model == 'vgg16': #Not available in MONAI
	base_model = torchvision.models.vgg16(pretrained=True)
	num_ftrs = base_model.classifier[6].in_features
	base_model.classifier[6] = nn.Linear(num_ftrs, 4)
	print("Training on VGG-16")

elif args.model == 'wide_resnet50_2': #Not available in MONAI
	base_model = torchvision.models.wide_resnet50_2(pretrained=True)
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Linear(num_ftrs, 4)
	print("Training on Wide Resnet 50X2")

elif args.model == 'densenet':
	base_model = densenet121(spatial_dims=2, in_channels=3, out_channels=4)
	print("Training on DenseNet121")

else:
	assert (False), "Give the model type"


base_model = base_model.to(device)
lr=0.000001
criterion = nn.CrossEntropyLoss()   # Categorical Cross Entropy Loss (the loss function)
optimizer = optim.Adam(base_model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-07, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


# Resuming from checkpoint
if args.resume:
	print('Resuming from checkpoint...')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.pth')
	base_model.load_state_dict(checkpoint['base_model'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']


############################## DEFINE MONAI TRAINERS 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger("ignite.engine.engine.SupervisedTrainer").setLevel(logging.WARNING)


post_transforms = Compose([
    AsDiscreted(keys="pred", argmax=True),
])


val_handlers = [
    StatsHandler(output_transform=lambda x: None),
    # TensorBoardStatsHandler(log_dir=base_direc+results_folder, output_transform=lambda x: None),
    # TensorBoardImageHandler(log_dir=base_direc+results_folder, batch_transform=lambda x: (x["image"], x["label"]), output_transform=lambda x: x["pred"]),
    CheckpointSaver(save_dir=base_direc+results_folder, save_dict={"base_model": base_model}, save_key_metric=True),
]

evaluator = SupervisedEvaluator(
    device=device,
    val_data_loader=testloader,
    network=base_model,
    inferer=SimpleInferer(),
    # post_transform=post_transforms,
    key_val_metric={"val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
    val_handlers=val_handlers,
    # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
    # amp=True if monai.config.get_torch_version_tuple() >= (1, 6) else False,
)

# Early stopping
def score_function(engine):
    val_acc = engine.state.key_val_metric['val_acc']
    return val_acc
    
def lr_score_function(engine):
    train_acc = engine.state.key_train_metric['train_acc']
    # print("!!!!!!!!!!!!", train_acc)
    print(engine.state)
    return (float)(train_acc)


train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        # LrScheduleHandler(lr_scheduler=scheduler, print_lr=True, step_transform=lambda x: lr_score_function),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        # TensorBoardStatsHandler(log_dir=base_direc+results_folder, tag_name="train_loss", output_transform=lambda x: x["loss"]),
        CheckpointSaver(save_dir=base_direc+results_folder, save_dict={"base_model": base_model, "optimizer": optimizer}, save_interval=2, epoch_level=True),
    ]


trainer = SupervisedTrainer(
    device=device,
    max_epochs=1000,
	train_data_loader=trainloader,
	network=base_model,
	optimizer=optimizer,
	loss_function=criterion,
    inferer=SimpleInferer(),
    # post_transform=post_transforms,
    key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
    train_handlers=train_handlers,
    # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP training
    # amp=True if monai.config.get_torch_version_tuple() >= (1, 6) else False,
    )

early_stopper = EarlyStopping(patience=20, score_function=stopping_fn_from_metric('val_acc'), trainer=trainer)
evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

# LR scheduler 
lr_scheduler_monai =  LrScheduleHandler(lr_scheduler=scheduler, print_lr=True, step_transform=lambda x: trainer.state.metrics["train_acc"])
# lr_scheduler.attach(trainer)

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,handler= lr_scheduler_monai)



##################################  CONVENTIONAL PYTORCH TRAINERS BUT DATASET FUNCTIONALITIES OF MONAI
def train(epoch):
	print('\nEpoch: %d' % epoch)
	batch_train_loss = 0
	correct = 0
	total = 0
	time_4 = time.time()
	for batch_idx, batch in enumerate(trainloader):
		# if(batch_idx>10):	break
		inputs  = batch['image']
		targets = batch['label']
		class_1=(targets == 0).sum()
		class_2=(targets == 1).sum()
		class_3=(targets == 2).sum()
		class_4	=(targets == 3).sum()
		base_model.train() 
		inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
		targets = targets.long()

		with torch.set_grad_enabled(True): # enable gradient while training
			outputs = base_model(inputs)
			loss = criterion(outputs, targets)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		batch_train_loss += loss.item()*inputs.size(0)
		_, predicted = outputs.max(1)

		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		progress_bar(batch_idx, len(trainloader), 'Batch Distri.: %.3f/%.3f/%.3f/%.3f | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
			 % (class_1,class_2,class_3,class_4, batch_train_loss/(total), 100.*correct/total, correct, total))

	train_accuracy = 100.*correct/total
	train_epoch_loss  = (float)(batch_train_loss/len(training_set))
	return train_epoch_loss, train_accuracy



# VALIDATON AFTER EACH EPOCH
def test(epoch, n_epochs_stop, testloader):
	global best_acc
	global min_val_loss
	global early_stop
	global epochs_no_improve
	base_model.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(testloader):
			# if(batch_idx>10):	break
			inputs  = batch['image']
			targets = batch['label']
			class_1=(targets == 0).sum()
			class_2=(targets == 1).sum()
			class_3=(targets == 2).sum()
			class_4	=(targets == 3).sum()
			inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
			outputs = base_model(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()*inputs.size(0)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
			progress_bar(batch_idx, len(testloader), 'Batch Distri.: %.3f/%.3f/%.3f/%.3f | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
			 % (class_1,class_2,class_3,class_4, test_loss/(total), 100.*correct/total, correct, total))


			
	valid_accuracy = 100.*correct/total
	# SAVE CHECKPOINT
	valid_epoch_loss  = (float)(test_loss/len(test_set))
	acc = 100.*correct/total
	if valid_epoch_loss < min_val_loss:
		print('Saving Checkpoint..')
		state = {
			'base_model': base_model.state_dict(),
			'acc': acc,
			'epoch': epoch,
			'loss':valid_epoch_loss
		}
		if not os.path.isdir(base_direc+results_folder+ 'checkpoint_'+str(lr)+replica_name):
			os.mkdir(base_direc+results_folder+ 'checkpoint_'+str(lr)+replica_name)
		torch.save(state, base_direc+results_folder+ 'checkpoint_'+str(lr)+replica_name+'/ckpt_check.pth')
		best_acc = acc
		print("Minimum val loss ", min_val_loss)
		print("new min val loss ", valid_epoch_loss)
		epochs_no_improve = 0
		min_val_loss = valid_epoch_loss
		print("epochs not improvement for early stop set to 0")

	else:
		epochs_no_improve += 1
		print("epochs not improvement for early stop ", epochs_no_improve)
		if epochs_no_improve == n_epochs_stop:
			print('Early stopping!' )
			early_stop = True

	return valid_epoch_loss, valid_accuracy





if args.monai_trainer:	trainer.run()
else:
	# CREATING THE EVALUATION FILE
	evaluationFile = open(base_direc+results_folder+ "Log"+str(lr) +replica_name+".txt" , "w")

	min_val_loss = np.Inf
	epochs_no_improve=0
	early_stop=False
	train_loss=[]
	valid_loss=[]
	n_epochs_stop=20
	evaluationFile.write("Epoch, training loss, validation loss, train accuracy, test accuracy" + "\n")
	for epoch in range(start_epoch, start_epoch+1000):  
		if(early_stop==False):  
			epoch_train_loss, epoch_train_acc = train(epoch)
			epoch_val_loss, epoch_val_acc = test(epoch, n_epochs_stop, testloader)
			train_loss.append(epoch_train_loss)
			valid_loss.append(epoch_val_loss)
			print("Epoch " + str(epoch) + " train loss = " + str(epoch_train_loss) + " valid loss = " + str(epoch_val_loss))
			evaluationFile.write(str(epoch) + ", " + str(epoch_train_loss) + ", " + str(epoch_val_loss) + ", " +str(epoch_train_acc) + ", " + str(epoch_val_acc) + "\n")
			scheduler.step(epoch_val_loss)
		else:
			print("Stopped due to early stopping")
			break


	evaluationFile.close()

	plt.plot(train_loss, label='Training loss')
	plt.plot(valid_loss, label='Validation loss')
	plt.xlabel("Epoch")
	plt.legend()
	plt.savefig(base_direc+results_folder+ 'loss_curves_'+str(lr)+replica_name+'.png')
	plt.show()










