import os

video_path = '/home/pirl/Documents/splited_action_data/'
#
# V = open('Violence.txt','w')
# nonV = open('NonViolence.txt', 'w')

fp = open(video_path+"/fastpass/FastPass.txt",'w')
r2l=open(video_path+'/right2left/Right2Left.txt',"w")
l2r = open(video_path+'/left2right/Left2Right.txt','w')


Pflist = os.listdir(video_path + 'fastpass/')

print(Pflist)
pf_file_list=[]

for filename in Pflist:
    pf_file_list.append(filename)
    pf_file_list.sort()


for filename in pf_file_list[1:]:
    name = filename.split('.')[0]
    if name=="FastPass":
        continue
    fp.write('fastpass/'+ name +' 1\n')

###################################################################
rllist = os.listdir(video_path + 'left2right/')


pf_file_list=[]

for filename in rllist:
    pf_file_list.append(filename)
    pf_file_list.sort()


for filename in pf_file_list[1:]:
    name = filename.split('.')[0]
    if name=="Left2Right":
        continue
    l2r.write('left2right/'+ name +' 2\n')

######################################################################
Pflist = os.listdir(video_path + 'right2left/')

pf_file_list = []

for filename in Pflist:
    pf_file_list.append(filename)
    pf_file_list.sort()

for filename in pf_file_list[1:]:
    name = filename.split('.')[0]
    if name=="Right2Left":
        continue
    r2l.write('right2left/' + name + ' 3\n')

######################################################################
fp.close()
r2l.close()
l2r.close()