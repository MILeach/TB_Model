with open('network.json', 'w') as f:
	f.write('{\n')
	f.write('"vertices": [\n')
	for i in range(0, 8000):
		f.write('{\n')
		f.write('"id": ' + str(i) +'\n')
		f.write('},\n')
	f.write('],\n')
	f.write('"edges": [\n')
	for i in range(0, 7999):
		f.write('{\n')
		f.write('"id": ' + str(i) +',\n')
		f.write('"source": ' + str(i) + ',\n')
		f.write('"destination": ' + str(i+1) + '\n')
		f.write('},\n')
	f.write('],\n')
	f.write('}')