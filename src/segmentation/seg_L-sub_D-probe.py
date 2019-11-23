import numpy as np
import pandas as pd 
import os
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import pickle
import argparse

datafolder = './data/raw/'
os.makedirs(datafolder, exist_ok=True)
os.makedirs('./data/seg_data', exist_ok=True)

csvname = {'pre':'_pre_infiniti.csv', 'main':'_main_infiniti.csv', 'post':'_post_infiniti.csv'}
labelname = ['empty','probe','target','target response']
labelindex = {'probe':1,'target':2,'target response':3}
labelcount = {0:0, 1:0}


def readalltrial(user, data_filename,  label_filename, plot=False, normalize=False, target_interval=10, probe_interval=3, check_interval=5 ,fs=256):
	
	# Signal and Label	
	sig_data = pd.read_csv(data_filename)
	eda = sig_data['SC-Pro/Flex - 1B'].values[1:]
	if normalize:
		eda = (eda-np.mean(eda))/np.std(eda)
	raw_label = sig_data['trigger'].values[1:]
	plot_label = np.zeros_like(raw_label)
	seg_data = {0:[],1:[]}
	

	# Self Label
	self_label_data = pd.read_csv(label_filename)
	self_label = np.array([d for d in self_label_data['Rating.RESP'] if not np.isnan(d)])
	mean = np.mean(self_label)
	self_label[self_label<mean] = 0
	self_label[self_label>=mean] = 1

	# Index
	all_label = np.array([labelindex[v] if v in labelindex else 0 for v in raw_label])
	normal_index = np.array([i for i, v in enumerate(raw_label) if v == 'normal'])
	target_index = np.array([i for i, v in enumerate(raw_label) if v == 'target'])
	probe_index = np.array([i for i, v in enumerate(raw_label) if v == 'probe'])	
	exp_label = sorted(np.concatenate((normal_index,target_index),axis=0))

	assert(len(self_label) == len(probe_index) == len(target_index))

	for n, i in enumerate(probe_index):
		if target_interval == 'all':
			stimuli = []
			for j in exp_label:
				if n == 0:
					if j < probe_index[n]:
						stimuli.append(j)
				else:
					if j < probe_index[n] and j > probe_index[n-1]:
						stimuli.append(j)
			start = stimuli[0]
			end = stimuli[-1]
		else:
			target_interval = int(target_interval)
			start = int(i-target_interval*fs if i-target_interval*fs > 0 else 0)
			end = i

		if self_label[n] == 1:
			labelcount[1] += 1
			seg_data[1].append(eda[start:end])
			plot_label[start:end] = labelindex['target response']
		else:
			labelcount[0] += 1
			seg_data[0].append(eda[start:end])
			plot_label[start:end] = labelindex['target']

	for i in probe_index:
		get = i-probe_interval*fs if i-probe_interval*fs > 0 else 0
		plot_label[int(get):i] = labelindex['probe']


	if plot:
		fig = plt.figure()
		color = ['w','green','pink','red']
		ax1 = fig.add_subplot(111)
		trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
		ax1.plot(eda)
		for i in range(len(np.unique(plot_label))):
			ax1.fill_between(range(len(plot_label)), 0, 1, where=plot_label==np.unique(plot_label)[i], facecolor=color[i], transform=trans, label=labelname[i])
		ax1.legend(loc='upper right', fontsize = 'small')
		ax1.set_xlabel('EDA',fontsize='medium')
		plt.show()
		plt.close()

	print('{}=== [0/1]:{}/{} mean:{} ==='.format(user, len(seg_data[0]),len(seg_data[1]), mean))

	return seg_data


def main(args):
	participant_list = list(range(1,61))
	participant_data = {}
	users = [6,7,8,13,14,18,26,31,32,40,42,43]
	# users = [1,2,3,4,5,6,7,8,9,13,14,15,16,17]
	labelfiles =  os.listdir(os.path.join(datafolder,'label/'))
	labelfiles.sort()
	for n, participant in enumerate(users):
		if participant in [3,4]:
			continue
		user = 'user'+ '{:02}'.format(participant)
		data_filename = datafolder+user+'/'+user+csvname['main']
		label_filename = os.path.join(datafolder,'label/',labelfiles[n])
		if os.path.exists(data_filename):
			seg_data = readalltrial(user, data_filename, label_filename, plot=args.plot, normalize=args.normalize, target_interval=args.time)
			participant_data[participant] = seg_data	
		#input()
	allcount = labelcount[0]+labelcount[1]
	print('Participant: {}'.format(len(participant_data.keys())))
	print('All seg count: {}'.format(allcount))
	print('0: {}({:.2f}%)'.format(labelcount[0],(labelcount[0]*100/allcount)))
	print('1: {}({:.2f}%)'.format(labelcount[1],(labelcount[1]*100/allcount)))
	pickle.dump(participant_data, open('./data/seg_data/'+args.outputfile+'.pkl', 'wb'))
	
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation.")
    parser.add_argument('outputfile', type=str, help='Directory to the outputfile.')
    parser.add_argument('--time', default=10, type=str)
    parser.add_argument('--plot', default=False, type=bool)
    parser.add_argument('--normalize', default=False, type=bool)
    args = parser.parse_args()
    main(args)