import datetime

def info(msg): 
    print(msg)
    f = open('training.log', 'a')
    # Write with time stamp 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(timestamp + ' ' + msg + '\n')




