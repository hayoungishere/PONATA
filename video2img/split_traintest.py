# Train:Test = 800:200 으로 랜덤분할
import random
fp = open("/home/pirl/Documents/splited_action_data/fastpass/FastPass.txt",'r')
r2l=open('/home/pirl/Documents/splited_action_data/right2left/Right2Left.txt',"r")
l2r = open('/home/pirl/Documents/splited_action_data/left2right/Left2Right.txt','r')

print(fp)

train = open('trainlist.txt', 'w')
test = open('testlist.txt', 'w')

fplist = fp.readlines()
random.shuffle(fplist)
r2llist = r2l.readlines()
random.shuffle(r2llist)
l2rlist = l2r.readlines()
random.shuffle(l2rlist)

for filename in fplist[0:17]:
    train.write(filename)

for filename in fplist[17:]:
    test.write(filename)

for filename in r2llist[0:17]:
    train.write(filename)

for filename in r2llist[17:]:
    test.write(filename)



for filename in l2rlist[0:17]:
    train.write(filename)

for filename in l2rlist[17:]:
    test.write(filename)



test.close()
train.close()

l2r.close()
r2l.close()
fp.close()
