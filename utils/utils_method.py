import os
import datetime


def printParOnFile(test_name, log_dir, par_list):

    assert isinstance(par_list, dict), "par_list as to be a dictionary"
    f=open(os.path.join(log_dir,test_name+".log"),'w+')
    f.write(test_name)
    f.write("\n")
    f.write(str(datetime.datetime.now().utcnow()))
    f.write("\n\n")
    for key, value in par_list.items():
        f.write(str(key)+": \t"+str(value))
        f.write("\n")

