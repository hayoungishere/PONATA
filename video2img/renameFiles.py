import os

home = '/home/pirl/Documents/action_data'

list = os.listdir(home)

print(list)
for d in list:
    inDirectory = os.listdir(home+"/"+d)
    #print(inDirectory)

    idx =1
    #print(home+'/'+d)
    for filename in inDirectory:
        #print(d,filename)
        dst = d+"_"+str(10000+idx)+"."+filename.split(".")[1]
        #print(dst)
        os.rename(os.path.join(home+'/'+d+"/", filename), os.path.join(home+"/"+d+"/", ''.join(dst)))

        #os.rename(filename, dst)
        idx+=1
    # for filename in os.listdir("/"+inDirectory):
    #
    #     dst = str(idx)+"."+filename.split(".")[1]
    #     print(filename, dst)
    #     os.rename(filename,dst)
    #     idx+=1


    inDirectory = os.listdir(home + "/" + d)10
    #print(inDirectory)
