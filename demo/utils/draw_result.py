from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import rcParams
import os.path as osp
import json

DATA_DIR = "/home/jing/Data/Projects/Mono-Following/codes/baselines/evaluation-on-JRDB/pytracking-based/results"

results = [
    "qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17-private-half.json",
    "tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.json",
    "bytetrack_yolox_x_crowdhuman_mot17-private-half.json",
    "siamese_rpn_r50_20e_otb100.json",
    "stark_st2_r50_50e_lasot.json"
]
NAMES      = ["QDTRACK", "TRACKTOR", "BYTETRACK", "SIAMESE", "STARK"]
LINESTYLES = ['-','-','-','-','-']
COLOURS    = ['g','y','m','c','r']
COLOURS2   = ['lightseagreen','lightseagreen','lightseagreen','lightseagreen','lightseagreen']
sequences = ['bytes-cafe-2019-02-07_0',
            'clark-center-2019-02-28_0', 'clark-center-2019-02-28_1', 'clark-center-intersection-2019-02-28_0', 
            'cubberly-auditorium-2019-04-22_0', 
            'forbes-cafe-2019-01-22_0', 
            'gates-159-group-meeting-2019-04-03_0', 
            'gates-ai-lab-2019-02-08_0', 
            'gates-basement-elevators-2019-01-17_1', 
            'gates-to-clark-2019-02-28_1', 
            'hewlett-packard-intersection-2019-01-24_0', 
            'huang-2-2019-01-25_0', 
            'huang-basement-2019-01-25_0', 
            'huang-lane-2019-02-12_0', 
            'jordan-hall-2019-04-22_0', 
            'memorial-court-2019-03-16_0', 
            'meyer-green-2019-03-16_0', 
            'nvidia-aud-2019-04-18_0', 
            'packard-poster-session-2019-03-20_0', 'packard-poster-session-2019-03-20_1', 'packard-poster-session-2019-03-20_2', 
            'stlc-111-2019-04-19_0', 
            'svl-meeting-gates-2-2019-04-08_0', 'svl-meeting-gates-2-2019-04-08_1', 
            'tressider-2019-03-16_0', 'tressider-2019-03-16_1', 
            'tressider-2019-04-26_2']

# ["Fully_visible", "Mostly_visible", "Severly_occluded", "Fully_occluded"]
errors_all = []
for index, result in enumerate(results):
    print(result)
    result_json = osp.join(DATA_DIR, result)
    js_pointer = open(result_json, 'r')
    result_dict = json.load(js_pointer)
    errors = []
    for sequence in sequences:
        result_seq = result_dict[sequence]
        for person_key in sorted(result_seq.keys()):
            result_person = result_seq[person_key]
            for image_key in sorted(result_person.keys()):
                result_img = result_seq[person_key][image_key]
                # For now, just evaluate the visible ones
                # if result_img["occlusion"] in ["Fully_visible", "Mostly_visible"]:
                if result_img["occlusion"] in ["Fully_visible"]:
                    distance = result_img["distance"]
                    if distance is not None:
                        errors.append(result_img["distance"])
                    else:
                        errors.append(200)
    errors_all.append(errors)
errors_all = np.array(errors_all)
print(errors_all.shape)
print(errors_all[:,0])
print(errors_all[:,1])
successful_rates = np.zeros([len(NAMES), 200])
for i in range(len(NAMES)):
    for j in range(200):
        successful_rates[i,j]  = errors_all[i][ np.where( errors_all[i] < j )].size/float(errors_all[i].size)
t = np.arange(0, successful_rates.shape[1], 1)

# plt.rcParams['text.usetex'] = True
# font = {'family': 'Times New Roman','size'   : 20}
# matplotlib.rc('font', **font)

plt.figure(figsize=(3.45,2))
# plt.figure(1)
successful_rates = successful_rates * 100 # use % as unit
x=np.arange(len(NAMES))
print(successful_rates[:,50])
barlist = plt.barh(x, successful_rates[:,50], align = "center", color=COLOURS2)
for _x, _y, _p in zip(successful_rates[:,50], x, successful_rates[:,50]):
   plt.text(_x+0.01, _y-0.1, "{:.1f}".format(_p), fontsize=6, style='italic')
plt.yticks(x, NAMES, fontsize=6)
plt.xlim([0, 110])
plt.xticks(fontsize=6)
plt.xlabel('Precision (\%)', fontsize=8)
# plt.title('Location Error Threshold = 50 pixels', fontsize=20)
plt.gcf().subplots_adjust(left=0.22)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().subplots_adjust(right=0.99)
plt.gcf().subplots_adjust(top=0.95)
plt.savefig(osp.join(DATA_DIR, "trackingBar.pdf"))
# plt.show()


plt.figure(figsize=(3.45,2))
plt.xlabel('Location Error Threshold (pixel)',fontsize=8)
plt.ylabel('Precision (\%)', fontsize=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.gcf().subplots_adjust(left=0.14)
plt.gcf().subplots_adjust(bottom=0.18)
plt.gcf().subplots_adjust(right=0.99)
plt.gcf().subplots_adjust(top=0.95)

lines = []
for i in range(len(NAMES)):
	line, = plt.plot(t, successful_rates[i], label=NAMES[i], linestyle=LINESTYLES[i], linewidth=1, c=COLOURS[i])
	lines.append(line)
plt.legend(loc=4,prop={'size':6})
plt.savefig(osp.join(DATA_DIR, "trackingPlot.pdf"))
# plt.show()	   
