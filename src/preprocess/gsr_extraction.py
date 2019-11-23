import os,csv
from gsr_utils import *
from gsr_feature import *
import warnings
from sys import argv
import _pickle as pk

PATH = argv[1]
NAME = argv[2]
NAME = NAME.strip()
plot = False
os.makedirs("./data/feature_data", exist_ok=True)

def writesignal(filename, participant ,signal):
    file = open(filename, 'a', encoding = 'big5')
    file.write(participant)
    for i in range(len(signal)):
        file.write(','+str(signal[i]))
    file.write('\n')
    file.close()

def feature_set(data,sig_type,fs):
    print('    get statistics_feature...')
    f1, name1 = statistics_feature(data)
    f2, name2 = statistics_feature(np.diff(data))
    f3, name3 = statistics_feature(np.diff(np.diff(data)))
    name2 = rename(name2, '1df.')
    name3 = rename(name3, '2df.')
    print('    get freq_feature...')
    f4, name4 = freq_feature(data,sig_type,fs)
    print('    get DWT_feature...')
    f5, name5 = DWT(data)
    print('    get entropy_feature...')
    f6, name6 = entropy_feature(data,match=2,scale=10)
    f = f1+f2+f3+f4+f5+f6
    name = name1+name2+name3+name4+name5+name6 
    name = rename(name, sig_type)
    return f, name

def main():
    writename = False
    pickle_in = open(PATH,"rb")
    file = pk.load(pickle_in)
    pickle_in.close()

    for participant in sorted(list(file.keys())):
        signal = file[participant]
        data = signal[0] + signal[1]
        label = [0]*len(signal[0]) + [1]*len(signal[1])

        for n, gsr_signal in enumerate(data):
            filename = str(participant)+'_'+str(n)
            print('user: {} | seg: {}'.format(participant,n))
            all_feature = []
            all_name = []
            FS = 256
            fs = 32
            gsr_signal = np.array(gsr_signal)
            gsr_filter = low_pass_filter(gsr_signal, fc=1.8, fs=FS, order=5)
            SC = gsr_filter
            SC = downsample(gsr_filter,fs=FS,nfs=fs)
            #for task in ['ori.','det.','win.','df.','bd.h.','bd.m.','bd.l.','lo.h.','lo.l.','CDA.','CVX.']:
            for task in ['lo.h.']:
                print('get '+task+'feature...')
                '''
                if task == 'ori.':
                    feature, name = feature_set(SC, 'o.', fs=fs)
                elif task == 'CDA.':
                    onset, peak, amp, driver, phasic, tonic = SCR_generate(SC,fs=fs,min_amplitude=0.5,task=task)    
                    #plotpeak(tonic+phasic,phasic,tonic,onset,peak,amp,filename,fs)
                    f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=fs)
                    print('  get phasic...')
                    f2, name2 = feature_set(phasic,'p.',fs=fs)
                    print('  get tonic...')
                    f3, name3 = feature_set(tonic,'t.',fs=fs)
                    feature = f1+f2+f3
                    name = name1+name2+name3
                    name = rename(name,task)
                elif task == 'CVX.':
                    onset, peak, amp, phasic, tonic = SCR_generate(SC,fs=fs,min_amplitude=0.3,task=task)    
                    #plotpeak(SC,phasic,tonic,onset,peak,amp,filename,fs)
                    f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=fs)
                    print('  get phasic...')
                    f2, name2 = feature_set(phasic,'p.',fs=fs)
                    print('  get tonic...')
                    f3, name3 = feature_set(tonic,'t.',fs=fs)
                    feature = f1+f2+f3
                    name = name1+name2+name3
                    name = rename(name,task)
                    '''

                onset, peak, amp, phasic = SCR_generate(SC,fs=fs,min_amplitude=0.5,task=task)
                tonic = SC - phasic
                orign = SC
                pksig = phasic
                f1, name1 = SCR_feature(phasic,onset,peak,amp,fs=fs)
                print('  get phasic...')
                f2, name2 = feature_set(phasic, 'p.',fs=fs)
                print('  get tonic...')
                f3, name3 = feature_set(tonic,'t.',fs=fs)
                feature = f1+f2+f3
                name = name1+name2+name3
                name = rename(name,task)
                all_feature += feature
                all_name += name    

                if plot:
                    plt.subplot(4,1,1,title='origin')
                    #plt.title('origin',loc='right')
                    plt.tight_layout()
                    plt.plot(gsr_signal)
                    plt.xticks([])
                    plt.subplot(4,1,2,title='preprocess')
                    plt.tight_layout()
                    #plt.title('preprocess',loc='right')
                    plt.plot(SC)
                    plt.xticks([])
                    plt.subplot(4,1,3,title='tonic')
                    plt.tight_layout()
                    #plt.title('tonic',loc='right')
                    plt.xticks([])
                    ts = np.linspace(0, (len(orign)-1)/fs,int(len(orign)),endpoint=False)

                    plt.scatter(ts,orign,s=2,c='b')
                    plt.scatter(ts,tonic,s=2,c='g')
                    plt.scatter(ts[peak],orign[peak],c='r',s=20)
                    plt.scatter(ts[onset],orign[onset],c='y',s=10)
                    plt.subplot(4,1,4,title='phasic')
                    plt.tight_layout()
                    #plt.title('phasic',loc='right')
                    ts = np.linspace(0, (len(pksig)-1)/fs,int(len(pksig)),endpoint=False)
                    plt.scatter(ts,pksig,s=2)
                    plt.scatter(ts[peak],pksig[peak],c='r',s=10)
                    plt.scatter(ts[onset],pksig[onset],c='y',s=10)
                    plt.show()
                    plt.close()     

            all_feature += [label[n]]
            all_name += ['label']
            print(len(all_feature),len(all_name))
            if writename == False:
                writesignal('./data/feature_data/{}.csv'.format(NAME),'',all_name) 
                writefile = open('./data/feature_data/feature_all.csv', 'w')
                for i,n in enumerate(all_name):
                    writefile.write(str(n)+'\n')
                writefile.close()
                writename = True    
            writesignal('./data/feature_data/{}.csv'.format(NAME),filename,all_feature)
            print('==========')
                
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()