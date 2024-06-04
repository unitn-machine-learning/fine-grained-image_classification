import os 

all_class = os.listdir('datasets/CompetitionData/train')
all_class.sort()
with open('datasets/CompetitionData/labels.txt', 'w') as file:
    i = 0
    for item in all_class:
        file.write(str(i) +' '+ str(int(item.split('_')[0])) + '\n')
        i+=1 