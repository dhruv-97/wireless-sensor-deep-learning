import csv

f = open("./WTD_upload/Toluene_200/L1/201106121329_board_setPoint_400V_fan_setPoint_000_mfc_setPoint_Toluene_200ppm_p1")
date = "20110612"
time = "1329"
csv = [[] for i in range(10)]
print(csv)
for line in f:
	
	data = line.split("\t")
	print(data, len(data))
	sec  =  int(data[0])
	fan = int(data[1])
	for j in range(10):
		k1 = 
		csv[j].append([date, time, sec, fan] + [data[i] for i in range(10*(j+1) + 2 - j, 10*(j+1) + 10 - j)])
	break
for j in range(10):
	with open('sensor' + j + '.csv', 'wb') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i,x in enumerate(csv[j]):
			filewriter.writerow(x)
10*(j+1) + 2 - j
